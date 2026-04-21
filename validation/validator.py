import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Ensure pathing works when executed from inside the validation folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.interpolation import batched_interp1d
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer
from utils.config import load_config
from utils.optimizer import refine_topology
from utils.seeding import set_seed

try:
    from validation.targets import TargetGenerator
except ImportError:
    from targets import TargetGenerator


class UnifiedValidator:
    def __init__(self, cfg_path="config.yaml"):
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join("..", cfg_path)
        self.cfg = load_config(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FIX: Define n_asp so it can be used in the baseline method
        self.n_asp = self.cfg['physics']['n_asperities']

        # 1. Establish Intensive Limits
        print("[Validator] Loading dataset for exact normalization limits...")
        data_path = self.cfg['data']['path']
        if not os.path.exists(data_path):
            data_path = os.path.join("..", data_path)

        data = torch.load(data_path, map_location=self.device)
        X = data["x"]

        self.MAX_P = data["p_star_max"]
        self.MAX_ALPHA = X[:, 1, :][X[:, 1, :] != -1.0].max().item()
        self.MAX_S = X[:, 2, :][X[:, 2, :] != -1.0].max().item()

        print(f"  > P_max: {self.MAX_P:.4e}, Alpha_max: {self.MAX_ALPHA:.4e}, S_max: {self.MAX_S:.4e}")

        # 2. Initialize Physics Engine & Model
        self.phys = AxisymmetricContactLayer(cfg=self.cfg).to(self.device)
        self.model = SurfaceInverseModel(self.cfg).to(self.device)

        model_name = self.cfg['model']['name']
        if not os.path.exists(model_name):
            model_name = os.path.join("..", model_name)
            
        print(f"[Validator] Loading model from {model_name}...")
        self.model.load_state_dict(torch.load(model_name, map_location=self.device))
        self.model.eval()

        self.gen = TargetGenerator(self.phys, self.cfg, self.device)

        # 3. Create the global P* grid
        self.steps = self.cfg['data']['n_steps']
        self.p_star_grid = torch.linspace(0, self.MAX_P, self.steps).to(self.device)

    # -------------------------------------------------------------------------
    # HELPER METHODS (Keeps the main logic clean and DRY)
    # -------------------------------------------------------------------------

    def _get_target(self, target_type):
        """Fetches the requested synthetic target, avoiding redundant if/else blocks."""
        if target_type == "linear": return self.gen.get_consistent_linear_coulomb()
        if target_type == "saturate": return self.gen.get_consistent_saturating()
        if target_type == "bilinear": return self.gen.get_consistent_bilinear()
        if target_type == "quadratic": return self.gen.get_consistent_quadratic()
        raise ValueError(f"Unknown target type: {target_type}")

    def _run_refinement(self, t_p, t_alpha, n_init, h_init, lock_n=False, **kwargs):
        return refine_topology(
            target_alpha=t_alpha, target_p=t_p,
            n_init=n_init, h_init=h_init,
            phys_engine=self.phys, p_star_grid=self.p_star_grid,
            t_w=self.gen.t_w, indentations=self.gen.indentations,
            lock_n=lock_n, **kwargs
        )

    def _align_to_grid(self, p, val):
        """Standardizes physical variables to the uniform P* grid for plotting."""
        aligned = batched_interp1d(self.p_star_grid, p.view(1, -1), val.view(1, -1), pad_value=-1.0)
        data = aligned[0].cpu().numpy()
        data[data == -1.0] = np.nan
        return data

    def prepare_nn_input(self, native_p, native_alpha, native_s):
        """Standardizes a native displacement-based curve for the P*-domain CNN."""
        aligned_alpha = batched_interp1d(self.p_star_grid, native_p.view(1, -1), native_alpha.view(1, -1), pad_value=-1.0)
        aligned_s = batched_interp1d(self.p_star_grid, native_p.view(1, -1), native_s.view(1, -1), pad_value=-1.0)

        norm_alpha = torch.where(aligned_alpha != -1.0, aligned_alpha / self.MAX_ALPHA, -1.0)
        norm_s = torch.where(aligned_s != -1.0, aligned_s / self.MAX_S, -1.0)
        norm_p = (self.p_star_grid / self.MAX_P).view(1, -1)

        return torch.stack([norm_p, norm_alpha, norm_s], dim=1)

    # -------------------------------------------------------------------------
    # MAIN VALIDATION LOGIC
    # -------------------------------------------------------------------------

    def get_test_set_indices_by_category(self):
        print("[Validator] Reconstructing Test Split to find unseen samples...")
        total_len = self.gen.total_samples
        train_len, val_len = int(0.8 * total_len), int(0.1 * total_len)
        
        _, _, test_ds = random_split(range(total_len), [train_len, val_len, total_len - train_len - val_len], generator=torch.Generator().manual_seed(42))

        categorized = {k: [] for k in self.gen.ranges}
        for idx in test_ds.indices:
            for cat, (start, end) in self.gen.ranges.items():
                if start <= idx < end:
                    categorized[cat].append(idx)
                    break
        return categorized

    def validate_on_test_set(self, refine=True):
        for category, indices in self.get_test_set_indices_by_category().items():
            if not indices: continue

            idx = np.random.choice(indices)
            idx += 1
            print(f"\nValidating {category.upper()} on Test Sample #{idx}...")

            # 1. Get Target Data (now explicitly using t_s and gt_n)
            t_p, t_a, t_s, gt_n, gt_h, _ = self.gen.get_custom_sample(idx, category)
            nn_input = self.prepare_nn_input(t_p, t_a, t_s)

            # 2. Zero-Shot Prediction
            with torch.no_grad():
                n_pred, h_pred = self.model(nn_input)
                # Re-sort and anchor to ensure physical consistency for the engine
                h_anchored = torch.sort(h_pred, dim=1)[0] - torch.sort(h_pred, dim=1)[0][:, 0:1]
                p_nn, a_nn, s_nn = self.phys(h_anchored, n_pred, self.gen.t_w, self.gen.indentations, k_steepness=1e5)

            # 3. Plotting with full 2x2 inputs
            if refine:
                n_ref, h_ref, p_ref, a_ref, s_ref = self._run_refinement(t_p, t_a, n_pred, h_pred)
                
                self.plot_triple_comparison(
                    t_p, t_a, t_s, gt_n, gt_h,
                    p_nn, a_nn, s_nn, n_pred, h_anchored,
                    p_ref, a_ref, s_ref, n_ref, h_ref,
                    title=f"Test: {category} (#{idx})"
                )
            else:
                self.plot_comparison(
                    t_p, t_a, t_s, gt_n, gt_h,
                    p_nn, a_nn, s_nn, n_pred, h_anchored,
                    title=f"Test: {category} (#{idx})"
                )

    def validate_designed(self, target_type="linear", refine=False):
        print(f"\n[Validator] Generating fresh synthetic target: {target_type}...")
        
        # 1. Fetch Target (t_s is now essential for the stiffness plot)
        t_p, t_a, t_s, title = self._get_target(target_type)
        nn_input = self.prepare_nn_input(t_p, t_a, t_s)

        # 2. Zero-Shot Prediction
        with torch.no_grad():
            n_pred, h_pred = self.model(nn_input)
            # Anchor to zero for physical consistency
            h_anchored = torch.sort(h_pred, dim=1)[0] - torch.sort(h_pred, dim=1)[0][:, 0:1]
            p_nn, a_nn, s_nn = self.phys(
                h_anchored, n_pred, self.gen.t_w, self.gen.indentations, k_steepness=1e5
            )

        # 3. Handle Refinement and plotting with the 2x2 grid signatures
        if refine:
            n_ref, h_ref, p_ref, a_ref, s_ref = self._run_refinement(t_p, t_a, n_pred, h_pred)
            
            self.plot_triple_comparison(
                t_p, t_a, t_s, None, None,                # No GT for designed targets
                p_nn, a_nn, s_nn, n_pred, h_anchored,     # NN predictions
                p_ref, a_ref, s_ref, n_ref, h_ref,        # Refined results
                title=f"Refined: {title}"
            )
        else:
            self.plot_comparison(
                t_p, t_a, t_s, None, None,                # No GT for designed targets
                p_nn, a_nn, s_nn, n_pred, h_anchored,     # NN predictions
                title=f"Unseen: {title}"
            )

    def validate_optimization_baseline(self, target_type="quadratic", n_starts=50):
        print(f"\n[Baseline] Tri-Comparison for {target_type} (CNN vs General Multi vs Hertz Multi)...")
        
        t_p, t_a, t_s, title = self._get_target(target_type)
        nn_input = self.prepare_nn_input(t_p, t_a, t_s)

        def compute_loss(pred_p, pred_a):
            t_al = batched_interp1d(self.p_star_grid, t_p.view(1, -1), t_a.view(1, -1), pad_value=-1.0)
            p_al = batched_interp1d(self.p_star_grid, pred_p.view(1, -1), pred_a.view(1, -1), pad_value=-1.0)
            mask = (t_al != -1.0).float()
            return torch.sum(torch.pow(p_al - t_al, 2) * mask) / (torch.sum(mask) + 1e-8)

        # --- Strategy A: CNN Initialization + General L-BFGS ---
        with torch.no_grad():
            n_cnn, h_cnn = self.model(nn_input)
        n_A, h_A, p_A, a_A, s_A = self._run_refinement(t_p, t_a, n_cnn, h_cnn)
        loss_A = compute_loss(p_A, a_A).item()
        print(f"  > [A] CNN Surrogate (n=var): {loss_A:.6f}")

        # --- Strategy B: Multi-Start General (n in [1,3]) ---
        best_loss_B, best_n_B, best_h_B = float('inf'), None, None
        for _ in range(n_starts):
            n_rand = torch.rand(1, self.n_asp).to(self.device) * 2.0 + 1.0
            h_rand = torch.sort(torch.rand(1, self.n_asp).to(self.device) * self.gen.max_d, dim=1)[0]
            n_prb, h_prb, p_prb, a_prb, _ = self._run_refinement(t_p, t_a, n_rand, h_rand)
            if (l := compute_loss(p_prb, a_prb).item()) < best_loss_B:
                best_loss_B, best_n_B, best_h_B = l, n_prb, h_prb
        n_B, h_B, p_B, a_B, s_B = self._run_refinement(t_p, t_a, best_n_B, best_h_B)
        loss_B = compute_loss(p_B, a_B).item()
        print(f"  > [B] Multi-Start General (n=var): {loss_B:.6f}")

        # --- Strategy C: Multi-Start Hertzian (n=2) ---
        best_loss_C, best_h_C = float('inf'), None
        for _ in range(n_starts):
            n_hertz = torch.ones(1, self.n_asp).to(self.device) * 2.0
            h_rand = torch.sort(torch.rand(1, self.n_asp).to(self.device) * self.gen.max_d, dim=1)[0]
            _, h_prb, p_prb, a_prb, _ = self._run_refinement(t_p, t_a, n_hertz, h_rand, lock_n=True)
            if (l := compute_loss(p_prb, a_prb).item()) < best_loss_C:
                best_loss_C, best_h_C = l, h_prb
        n_C, h_C, p_C, a_C, s_C = self._run_refinement(t_p, t_a, torch.ones_like(n_hertz)*2.0, best_h_C, lock_n=True)
        loss_C = compute_loss(p_C, a_C).item()
        print(f"  > [C] Multi-Start Hertz (n=2):   {loss_C:.6f}")

        # --- PLOTTING ---
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        p_axis = self.p_star_grid.cpu().numpy()
        max_p_val = t_p.max().item()
        
        C_GT, C_CNN, C_GEN, C_HERTZ = '#333333', '#0072B2', '#009E73', '#D55E00' # Black, Teal, Green, Red

        def style_ax(ax):
            ax.grid(True, alpha=0.15); ax.tick_params(direction='in', top=True, right=True)
            ax.set_xlim(0, max_p_val * 1.1)

        def plot_stem(ax, x, y, color, marker, label):
            markerline, stemlines, _ = ax.stem(x, y, basefmt=" ")
            plt.setp(markerline, color=color, marker=marker, markersize=6, label=label)
            plt.setp(stemlines, color=color, lw=1.5, alpha=0.5)

        # [0,0] Area
        axs[0,0].plot(p_axis, self._align_to_grid(t_p, t_a), color=C_GT, lw=3, label="Target")
        axs[0,0].plot(p_axis, self._align_to_grid(p_C, a_C), color=C_HERTZ, ls='-.', lw=2, label=f"Hertz (n=2)")
        axs[0,0].plot(p_axis, self._align_to_grid(p_B, a_B), color=C_GEN, ls='--', lw=2, label=f"MS General (n=var)")
        axs[0,0].plot(p_axis, self._align_to_grid(p_A, a_A), color=C_CNN, ls=':', lw=4, label="CNN Surrogate")
        axs[0,0].set(title="Baseline: Area vs Load", ylabel="Contact Fraction α")
        style_ax(axs[0,0]); axs[0,0].legend()

        # [1,0] Stiffness
        axs[1,0].plot(p_axis, self._align_to_grid(t_p, t_s), color=C_GT, lw=3)
        axs[1,0].plot(p_axis, self._align_to_grid(p_C, s_C), color=C_HERTZ, ls='-.', lw=2)
        axs[1,0].plot(p_axis, self._align_to_grid(p_B, s_B), color=C_GEN, ls='--', lw=2)
        axs[1,0].plot(p_axis, self._align_to_grid(p_A, s_A), color=C_CNN, ls=':', lw=4)
        axs[1,0].set(title="Baseline: Stiffness vs Load", xlabel="Nominal Pressure P*", ylabel="Stiffness S*")
        style_ax(axs[1,0])

        idx = np.arange(self.n_asp)
        s_A, s_B, s_C = [torch.argsort(h[0]).cpu().numpy() for h in [h_A, h_B, h_C]]

        # [0,1] Heights
        plot_stem(axs[0,1], idx - 0.2, h_C[0][s_C].cpu().numpy(), C_HERTZ, 'D', 'Hertz')
        plot_stem(axs[0,1], idx, h_B[0][s_B].cpu().numpy(), C_GEN, '^', 'MS General')
        plot_stem(axs[0,1], idx + 0.2, h_A[0][s_A].cpu().numpy(), C_CNN, 's', 'CNN Surrogate')
        axs[0,1].set(title="Optimized Heights", ylabel="h [m]"); axs[0,1].legend()

        # [1,1] Exponents
        axs[1,1].axhline(2.0, color=C_HERTZ, ls='-', alpha=0.3)
        plot_stem(axs[1,1], idx - 0.2, n_C[0][s_C].detach().numpy(), C_HERTZ, 'D', 'Hertz')
        plot_stem(axs[1,1], idx, n_B[0][s_B].detach().numpy(), C_GEN, '^', 'MS General')
        plot_stem(axs[1,1], idx + 0.2, n_A[0][s_A].detach().numpy(), C_CNN, 's', 'CNN Surrogate')
        axs[1,1].set(title="Optimized Exponents", xlabel="Asperity Index", ylabel="n [-]", ylim=(0.8, 3.2))
        axs[1,1].legend(loc='lower right', fontsize='small')

        plt.tight_layout()
        import os; os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/baseline_3way_{target_type}.png", dpi=150); plt.close()

    # -------------------------------------------------------------------------
    # PLOTTING LOGIC
    # -------------------------------------------------------------------------
    def plot_test_set_overview(self, save_path="plots/fig2_test_set_overview.png"):
        """
        Generates a publication-ready 2x3 grid showing one random unseen 
        test sample from each of the 6 topological categories.
        """
        print("\n[Validator] Generating Dataset Overview Figure from Test Set...")
        categorized_indices = self.get_test_set_indices_by_category()

        # -------------------------------------------------------------------
        # 1. PUBLICATION-GRADE PLOTTING SETUP
        # -------------------------------------------------------------------

        fig, axs = plt.subplots(2, 3, figsize=(7, 4), sharex=True, sharey=True)
        axs = axs.flatten()
        
        C_DATA = '#2F4F4F' # Dark Slate Gray
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        p_axis = self.p_star_grid.cpu().numpy()

        # -------------------------------------------------------------------
        # 2. FETCH AND PLOT DATA
        # -------------------------------------------------------------------
        for i, (category, indices) in enumerate(categorized_indices.items()):
            if i >= 6: break # Safety limit for 2x3 grid
            
            ax = axs[i]
            
            if not indices:
                ax.text(0.5, 0.5, f"No Test Data:\n{category}", ha='center', va='center')
                continue

            # Pick a random unseen sample (+1 if your generator expects 1-based indexing)
            idx = np.random.choice(indices) + 1 
            
            # Fetch Ground Truth Data
            t_p, t_a, t_s, _, _, _ = self.gen.get_custom_sample(idx, category)
            
            # Align to global grid for clean, consistent plotting
            a_aligned = self._align_to_grid(t_p, t_a)

            # Plot the curve
            ax.plot(p_axis, a_aligned, color=C_DATA, linewidth=3, solid_capstyle='round')
            
            # Grid and spines
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Title and Panel Label
            ax.set_title(category.replace("_", " ").title(), fontsize=14, pad=12, fontweight='bold')
            ax.text(0.05, 0.90, panel_labels[i], transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='top')
            
            # Formatting
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            ax.set_xlim(0, 0.0075)
            
            if i >= 3:
                ax.set_xlabel('$P^*$', fontsize=13)
            if i % 3 == 0:
                ax.set_ylabel(r'$\alpha$', fontsize=13)

        # -------------------------------------------------------------------
        # 3. SAVE AND EXPORT
        # -------------------------------------------------------------------
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25, wspace=0.1) 
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)


    def plot_comparison(self, t_p, t_a, t_s, gt_n, gt_h, p_nn, a_nn, s_nn, n_nn, h_nn, title):
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Colorblind-friendly palette
        C_GT = '#333333'   # Dark Charcoal
        C_NN = '#0072B2'   # Teal / Sky Blue

        p_axis = self.p_star_grid.cpu().numpy()
        max_p_val = t_p.max().item()

        # Helper to format axes cleanly
        def style_ax(ax):
            ax.grid(True, alpha=0.15, color='gray')
            ax.tick_params(axis='both', direction='in', top=True, right=True)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            ax.set_xlim(0, max_p_val * 1.1)

        # Helper to plot stems
        def plot_stem(ax, x, y, color, marker, label):
            markerline, stemlines, baseline = ax.stem(x, y, basefmt=" ")
            plt.setp(markerline, color=color, marker=marker, markersize=7)
            plt.setp(stemlines, color=color, linestyle='-', linewidth=2, alpha=0.6)
            markerline.set_label(label)

        # --- [0, 0] Contact Area ---
        axs[0, 0].plot(p_axis, self._align_to_grid(t_p, t_a), color=C_GT, lw=3, label="Target")
        axs[0, 0].plot(p_axis, self._align_to_grid(p_nn, a_nn), color=C_NN, linestyle='--', lw=2.5, label="Prediction")
        axs[0, 0].set(title="Contact Area vs Load (α vs P*)", xlabel="Nominal Pressure P*", ylabel="Contact Fraction α")
        style_ax(axs[0, 0])
        axs[0, 0].legend()

        # --- [1, 0] Topological Stiffness ---
        axs[1, 0].plot(p_axis, self._align_to_grid(t_p, t_s), color=C_GT, lw=3, label="Target")
        axs[1, 0].plot(p_axis, self._align_to_grid(p_nn, s_nn), color=C_NN, linestyle='--', lw=2.5, label="Prediction")
        axs[1, 0].set(title="Topological Stiffness (S* vs P*)", xlabel="Nominal Pressure P*", ylabel="Stiffness S*")
        style_ax(axs[1, 0])
        axs[1, 0].legend()

        # --- Sort Data by Height ---
        idx = np.arange(h_nn.shape[1])
        sorted_idx_nn = torch.argsort(h_nn[0]).cpu().numpy()
        nn_h_np = h_nn[0][sorted_idx_nn].detach().cpu().numpy()
        nn_n_np = n_nn[0][sorted_idx_nn].detach().cpu().numpy()

        if gt_h is not None:
            sorted_idx_gt = torch.argsort(gt_h[0]).cpu().numpy()
            gt_h_np = gt_h[0][sorted_idx_gt].cpu().numpy()
            gt_n_np = gt_n[0][sorted_idx_gt].cpu().numpy()

        # --- [0, 1] Height Distribution (Stem Plot) ---
        if gt_h is not None:
            plot_stem(axs[0, 1], idx - 0.1, gt_h_np, C_GT, 'o', 'Ground Truth')
            plot_stem(axs[0, 1], idx + 0.1, nn_h_np, C_NN, 's', 'Prediction')
        else:
            plot_stem(axs[0, 1], idx, nn_h_np, C_NN, 's', 'Prediction')

        axs[0, 1].set(title="Predicted Topography Structure (Heights)", xlabel="Asperity Index", ylabel="Height Offset h [m]")
        axs[0, 1].grid(True, alpha=0.15); axs[0, 1].tick_params(direction='in')
        axs[0, 1].legend()

        # --- [1, 1] Exponent Distribution (Stem Plot) ---
        axs[1, 1].axhline(1.0, color='gray', linestyle=':', alpha=0.8, label="Cone (n=1)")
        axs[1, 1].axhline(2.0, color='gray', linestyle='--', alpha=0.5, label="Sphere (n=2)")
        axs[1, 1].axhline(3.0, color='gray', linestyle='-.', alpha=0.5, label="Cubic (n=3)")
        
        if gt_h is not None:
            plot_stem(axs[1, 1], idx - 0.1, gt_n_np, C_GT, 'o', 'Ground Truth')
            plot_stem(axs[1, 1], idx + 0.1, nn_n_np, C_NN, 's', 'Prediction')
        else:
            plot_stem(axs[1, 1], idx, nn_n_np, C_NN, 's', 'Prediction')

        axs[1, 1].set(title="Shape Exponent Distribution (n)", xlabel="Asperity Index", ylabel="Shape Exponent n [-]")
        axs[1, 1].set_ylim(0.8, 3.2)
        axs[1, 1].grid(True, alpha=0.15); axs[1, 1].tick_params(direction='in')
        axs[1, 1].legend(loc='lower right', fontsize='small')

        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else title.split(" ")[0].lower()
        plt.tight_layout()
        plt.savefig(f"plots/val_{sname}.png", dpi=150)
        plt.close()

    def plot_triple_comparison(self, t_p, t_a, t_s, gt_n, gt_h, p_nn, a_nn, s_nn, n_nn, h_nn, p_ref, a_ref, s_ref, n_ref, h_ref, title):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        
        # Colorblind-friendly palette
        C_GT = '#333333'   # Dark Charcoal
        C_NN = '#0072B2'   # Teal / Sky Blue
        C_REF = '#D55E00'  # Burnt Orange / Vermilion

        p_axis = self.p_star_grid.cpu().numpy()
        max_p_val = t_p.max().item()

        def style_ax(ax):
            ax.grid(True, alpha=0.15, color='gray')
            ax.tick_params(axis='both', direction='in', top=True, right=True)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            ax.set_xlim(0, max_p_val * 1.1)

        def plot_stem(ax, x, y, color, marker, label):
            markerline, stemlines, baseline = ax.stem(x, y, basefmt=" ")
            plt.setp(markerline, color=color, marker=marker, markersize=7)
            plt.setp(stemlines, color=color, linestyle='-', linewidth=2, alpha=0.6)
            markerline.set_label(label)

        # --- [0, 0] Contact Area ---
        axs[0, 0].plot(p_axis, self._align_to_grid(t_p, t_a), color=C_GT, lw=3, label="Target")
        axs[0, 0].plot(p_axis, self._align_to_grid(p_nn, a_nn), color=C_NN, linestyle='--', lw=2.5, label="Zero-Shot (NN)")
        axs[0, 0].plot(p_axis, self._align_to_grid(p_ref, a_ref), color=C_REF, linestyle=':', lw=3.5, label="Refined (Opt)")
        axs[0, 0].set(title="Contact Area vs Load (α vs P*)", xlabel="Nominal Pressure P*", ylabel="Contact Fraction α")
        style_ax(axs[0, 0])
        axs[0, 0].legend()

        # --- [1, 0] Topological Stiffness ---
        axs[1, 0].plot(p_axis, self._align_to_grid(t_p, t_s), color=C_GT, lw=3, label="Target")
        axs[1, 0].plot(p_axis, self._align_to_grid(p_nn, s_nn), color=C_NN, linestyle='--', lw=2.5, label="Zero-Shot (NN)")
        axs[1, 0].plot(p_axis, self._align_to_grid(p_ref, s_ref), color=C_REF, linestyle=':', lw=3.5, label="Refined (Opt)")
        axs[1, 0].set(title="Topological Stiffness (S* vs P*)", xlabel="Nominal Pressure P*", ylabel="Stiffness S*")
        style_ax(axs[1, 0])
        axs[1, 0].legend()

        # --- Sort Data by Height ---
        idx = np.arange(h_nn.shape[1])
        sorted_idx_nn = torch.argsort(h_nn[0]).cpu().numpy()
        nn_h_np = h_nn[0][sorted_idx_nn].detach().cpu().numpy()
        nn_n_np = n_nn[0][sorted_idx_nn].detach().cpu().numpy()
        
        sorted_idx_ref = torch.argsort(h_ref[0]).cpu().numpy()
        ref_h_np = h_ref[0][sorted_idx_ref].detach().cpu().numpy()
        ref_n_np = n_ref[0][sorted_idx_ref].detach().cpu().numpy()

        if gt_h is not None:
            sorted_idx_gt = torch.argsort(gt_h[0]).cpu().numpy()
            gt_h_np = gt_h[0][sorted_idx_gt].cpu().numpy()
            gt_n_np = gt_n[0][sorted_idx_gt].cpu().numpy()

        # --- [0, 1] Height Distribution (Stem Plot) ---
        if gt_h is not None:
            plot_stem(axs[0, 1], idx - 0.15, gt_h_np, C_GT, 'o', 'Ground Truth')
            plot_stem(axs[0, 1], idx, nn_h_np, C_NN, 's', 'NN Pred')
            plot_stem(axs[0, 1], idx + 0.15, ref_h_np, C_REF, '^', 'Refined')
        else:
            plot_stem(axs[0, 1], idx - 0.1, nn_h_np, C_NN, 's', 'NN Pred')
            plot_stem(axs[0, 1], idx + 0.1, ref_h_np, C_REF, '^', 'Refined')

        axs[0, 1].set(title="Predicted Topography Structure (Heights)", xlabel="Asperity Index", ylabel="Height Offset h [m]")
        axs[0, 1].grid(True, alpha=0.15); axs[0, 1].tick_params(direction='in')
        axs[0, 1].legend()

        # --- [1, 1] Exponent Distribution (Stem Plot) ---
        axs[1, 1].axhline(1.0, color='gray', linestyle=':', alpha=0.8, label="Cone (n=1)")
        axs[1, 1].axhline(2.0, color='gray', linestyle='--', alpha=0.5, label="Sphere (n=2)")
        axs[1, 1].axhline(3.0, color='gray', linestyle='-.', alpha=0.5, label="Cubic (n=3)")

        if gt_h is not None:
            plot_stem(axs[1, 1], idx - 0.15, gt_n_np, C_GT, 'o', 'Ground Truth')
            plot_stem(axs[1, 1], idx, nn_n_np, C_NN, 's', 'NN Pred')
            plot_stem(axs[1, 1], idx + 0.15, ref_n_np, C_REF, '^', 'Refined')
        else:
            plot_stem(axs[1, 1], idx - 0.1, nn_n_np, C_NN, 's', 'NN Pred')
            plot_stem(axs[1, 1], idx + 0.1, ref_n_np, C_REF, '^', 'Refined')

        axs[1, 1].set(title="Shape Exponent Distribution (n)", xlabel="Asperity Index", ylabel="Shape Exponent n [-]")
        axs[1, 1].set_ylim(0.8, 3.2)
        axs[1, 1].grid(True, alpha=0.15); axs[1, 1].tick_params(direction='in')
        
        # Put legend in a good spot so it doesn't cover data
        axs[1, 1].legend(loc='lower right', fontsize='small')

        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else "sample"
        plt.tight_layout()
        plt.savefig(f"plots/val_test_{sname}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    val = UnifiedValidator("config.yaml")
    set_seed(17)
    
    val.plot_test_set_overview()
    # val.validate_on_test_set()
    # val.validate_designed(target_type="linear", refine=True)
    # val.validate_designed(target_type="saturate", refine=True)
    # val.validate_designed(target_type="bilinear", refine=True)
    
    # val.validate_optimization_baseline(target_type="saturate")
    # val.validate_optimization_baseline(target_type="bilinear")
    # val.validate_optimization_baseline(target_type="linear")
    # val.validate_optimization_baseline(target_type="quadratic")