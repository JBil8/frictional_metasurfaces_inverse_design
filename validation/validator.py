import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

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
        raise ValueError(f"Unknown target type: {target_type}")

    def _run_refinement(self, t_p, t_alpha, n_init, h_init, **kwargs):
        """Wraps the external optimizer utility to automatically pass class context."""
        return refine_topology(
            target_alpha=t_alpha, target_p=t_p,
            n_init=n_init, h_init=h_init,
            phys_engine=self.phys, p_star_grid=self.p_star_grid,
            t_w=self.gen.t_w, indentations=self.gen.indentations,
            **kwargs
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

    def validate_optimization_baseline(self, target_type="bilinear", n_starts=50):
        print(f"\n[Baseline] Comparing CNN vs Multi-Start ({n_starts} guesses) for {target_type}...")
        
        # 1. Fetch Target
        t_p, t_a, t_s, title = self._get_target(target_type)
        nn_input = self.prepare_nn_input(t_p, t_a, t_s)

        def compute_loss(pred_p, pred_a):
            t_al = batched_interp1d(self.p_star_grid, t_p.view(1, -1), t_a.view(1, -1), pad_value=-1.0)
            p_al = batched_interp1d(self.p_star_grid, pred_p.view(1, -1), pred_a.view(1, -1), pad_value=-1.0)
            mask = (t_al != -1.0).float()
            return torch.sum(torch.pow(p_al - t_al, 2) * mask) / (torch.sum(mask) + 1e-8)

        # 2. Strategy A: CNN Initialization (Teal)
        with torch.no_grad():
            n_cnn, h_cnn = self.model(nn_input)
            
        n_A, h_A, p_A, a_A, s_A = self._run_refinement(t_p, t_a, n_cnn, h_cnn)
        loss_A = compute_loss(p_A, a_A).item()
        print(f"  > [CNN] Alpha Error: {loss_A:.6f}")

        # 3. Strategy B: Multi-Start (Green)
        print(f"  > Strategy B: Multi-Start ({n_starts} random guesses)...")
        best_loss_B, best_n, best_h = float('inf'), None, None
        
        for _ in range(n_starts):
            n_rand = torch.rand(1, self.n_asp).to(self.device) * 2.0 + 1.0
            h_rand = torch.rand(1, self.n_asp).to(self.device) * self.gen.max_d
            h_rand = torch.sort(h_rand, dim=1)[0]
            h_rand = h_rand - h_rand[:, 0:1]

            # Short probe
            n_prb, h_prb, p_prb, a_prb, _ = self._run_refinement(t_p, t_a, n_rand, h_rand)
            
            if (l := compute_loss(p_prb, a_prb).item()) < best_loss_B:
                best_loss_B, best_n, best_h = l, n_prb, h_prb

        # Final refinement for best random guess
        n_B, h_B, p_B, a_B, s_B = self._run_refinement(t_p, t_a, best_n, best_h)
        loss_B = compute_loss(p_B, a_B).item()
        
        print(f"  > [Multi-Start] Alpha Error: {loss_B:.6f}")
        print(f"  > CNN is {loss_B / (loss_A + 1e-12):.1f}x more accurate.")

        # --- 2x2 PLOTTING BLOCK ---
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        p_axis = self.p_star_grid.cpu().numpy()
        max_p_val = t_p.max().item()
        
        C_GT, C_CNN, C_MS = '#333333', '#0072B2', '#009E73' # Black, Teal, Green

        def style_ax(ax):
            ax.grid(True, alpha=0.15); ax.tick_params(direction='in', top=True, right=True)
            ax.set_xlim(0, max_p_val * 1.1)

        def plot_stem(ax, x, y, color, marker, label):
            markerline, stemlines, _ = ax.stem(x, y, basefmt=" ")
            plt.setp(markerline, color=color, marker=marker, markersize=7, label=label)
            plt.setp(stemlines, color=color, lw=2, alpha=0.6)

        # [0,0] Area
        axs[0,0].plot(p_axis, self._align_to_grid(t_p, t_a), color=C_GT, lw=3, label="Target")
        axs[0,0].plot(p_axis, self._align_to_grid(p_B, a_B), color=C_MS, ls='--', lw=2, label=f"Multi-Start ({n_starts}x)")
        axs[0,0].plot(p_axis, self._align_to_grid(p_A, a_A), color=C_CNN, ls=':', lw=4, label="CNN Init (1x)")
        axs[0,0].set(title="Baseline: Area vs Load", ylabel="Contact Fraction α")
        style_ax(axs[0,0]); axs[0,0].legend()

        # [1,0] Stiffness
        axs[1,0].plot(p_axis, self._align_to_grid(t_p, t_s), color=C_GT, lw=3)
        axs[1,0].plot(p_axis, self._align_to_grid(p_B, s_B), color=C_MS, ls='--', lw=2)
        axs[1,0].plot(p_axis, self._align_to_grid(p_A, s_A), color=C_CNN, ls=':', lw=4)
        axs[1,0].set(title="Baseline: Stiffness vs Load", xlabel="Nominal Pressure P*", ylabel="Stiffness S*")
        style_ax(axs[1,0])

        # Data Prep
        idx = np.arange(self.n_asp)
        s_idx_A = torch.argsort(h_A[0]).cpu().numpy()
        s_idx_B = torch.argsort(h_B[0]).cpu().numpy()

        # [0,1] Heights
        plot_stem(axs[0,1], idx - 0.1, h_B[0][s_idx_B].cpu().numpy(), C_MS, '^', 'Multi-Start')
        plot_stem(axs[0,1], idx + 0.1, h_A[0][s_idx_A].cpu().numpy(), C_CNN, 's', 'CNN Init')
        axs[0,1].set(title="Optimized Heights", ylabel="h [m]"); axs[0,1].legend()

        # [1,1] Exponents
        axs[1,1].axhline(1.0, color='gray', ls=':', alpha=0.5); axs[1,1].axhline(2.0, color='gray', ls='--', alpha=0.5)
        plot_stem(axs[1,1], idx - 0.1, n_B[0][s_idx_B].cpu().numpy(), C_MS, '^', 'Multi-Start')
        plot_stem(axs[1,1], idx + 0.1, n_A[0][s_idx_A].cpu().numpy(), C_CNN, 's', 'CNN Init')
        axs[1,1].set(title="Optimized Exponents", xlabel="Asperity Index", ylabel="n [-]", ylim=(0.8, 3.2))
        axs[1,1].legend(loc='lower right', fontsize='small')

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/baseline_2x2_{target_type}.png", dpi=150); plt.close()

    # -------------------------------------------------------------------------
    # PLOTTING LOGIC
    # -------------------------------------------------------------------------

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
    
    val.validate_on_test_set()
    val.validate_designed(target_type="linear", refine=True)
    val.validate_designed(target_type="saturate", refine=True)
    val.validate_designed(target_type="bilinear", refine=True)
    
    val.validate_optimization_baseline(target_type="saturate")
    val.validate_optimization_baseline(target_type="bilinear")
    val.validate_optimization_baseline(target_type="linear")