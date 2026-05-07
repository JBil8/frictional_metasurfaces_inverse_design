import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

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

from utils.seeding import set_seed
from utils.optimizer import refine_topology
from utils.config import load_config
from physics.differentiable import AxisymmetricContactLayer
from ml_models.model_mlp import SurfaceInverseModel
from utils.interpolation import batched_interp1d

try:
    from validation.targets import TargetGenerator
except ImportError:
    from targets import TargetGenerator


class UnifiedValidator:
    def __init__(self, cfg_path="config.yaml"):
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join("..", cfg_path)
        self.cfg = load_config(cfg_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.n_asp = self.cfg['physics']['n_asperities']
        self.steps = self.cfg['data']['n_steps']

        self.gamma_max = self.cfg['physics']['gamma_max']
        self.gamma_min = self.cfg['physics']['gamma_min']

        # 1. Establish Intensive Limits
        print("[Validator] Loading dataset for exact normalization limits...")
        data_path = self.cfg['data']['path']
        if not os.path.exists(data_path):
            data_path = os.path.join("..", data_path)

        data = torch.load(data_path, map_location=self.device)
        self.MAX_P = data.get("p_star_max_global", 1.0)
        print(f"  > Dataset Global P_max established at: {self.MAX_P:.4e}")

        # 2. Initialize Physics Engine & Model
        self.phys = AxisymmetricContactLayer(cfg=self.cfg).to(self.device)
        self.model = SurfaceInverseModel(self.cfg).to(self.device)

        model_name = self.cfg['model']['name']
        if not os.path.exists(model_name):
            model_name = os.path.join("..", model_name)

        print(f"[Validator] Loading model from {model_name}...")
        self.model.load_state_dict(torch.load(
            model_name, map_location=self.device))
        self.model.eval()

        self.gen = TargetGenerator(self.phys, self.cfg, self.device)

        # 3. Create the Normalized Grid [0, 1] for Neural Network inference
        self.p_hat_grid = torch.linspace(0, 1.0, self.steps).to(self.device)

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _get_target(self, target_type):
        if target_type == "linear":
            return self.gen.get_consistent_linear_coulomb()
        if target_type == "saturate":
            return self.gen.get_consistent_saturating()
        if target_type == "bilinear":
            return self.gen.get_consistent_bilinear()
        if target_type == "quadratic":
            return self.gen.get_consistent_quadratic()
        raise ValueError(f"Unknown target type: {target_type}")

    def _run_refinement(self, t_p, t_alpha, n_init, h_init, lock_n=False, **kwargs):
        # Note: L-BFGS optimizer requires the global unscaled p_grid if your old optimizer
        # code hasn't been updated to use the hat notation yet. We pass a mock grid here
        # or you can update `refine_topology` to use absolute comparisons.
        dummy_p_grid = torch.linspace(
            0, self.MAX_P, self.steps).to(self.device)
        return refine_topology(
            target_alpha=t_alpha, target_p=t_p,
            n_init=n_init, h_init=h_init,
            phys_engine=self.phys, p_star_grid=dummy_p_grid,
            t_w=self.gen.t_w, indentations=self.gen.indentations,
            gamma_min=self.gamma_min, gamma_max=self.gamma_max,
            lock_n=lock_n, **kwargs
        )
    
    def _evaluate_baseline(self, target_type="saturate", n_starts=50):
        print(f"\n[Evaluating] {target_type.capitalize()} (CNN vs General Multi vs Hertz Multi)...")

        t_p, t_a, t_s, title = self._get_target(target_type)
        x_arr, x_scal_log = self.prepare_nn_input(t_p, t_a, t_s)

        def compute_loss(pred_p, pred_a):
            pred_a_aligned = batched_interp1d(
                t_p[0], pred_p, pred_a, pad_value=pred_a[0, -1].item())
            return torch.mean((pred_a_aligned - t_a)**2)

        # --- A: MLP Surrogate ---
        with torch.no_grad():
            n_cnn, h_cnn = self.model(x_arr, x_scal_log)
        n_A, h_A, p_A, a_A, s_A = self._run_refinement(t_p, t_a, n_cnn, h_cnn)

        # --- B: Multi-Start General ---
        best_loss_B, best_n_B, best_h_B = float('inf'), None, None
        gamma_range = self.gamma_max - self.gamma_min
        
        for _ in range(n_starts):
            n_rand = self.gamma_min + torch.rand(1, self.n_asp).to(self.device) * gamma_range
            h_rand = torch.sort(torch.rand(1, self.n_asp).to(self.device) * self.gen.max_d, dim=1)[0]
            
            n_prb, h_prb, p_prb, a_prb, _ = self._run_refinement(t_p, t_a, n_rand, h_rand)
            if (l := compute_loss(p_prb, a_prb).item()) < best_loss_B:
                best_loss_B, best_n_B, best_h_B = l, n_prb, h_prb
        n_B, h_B, p_B, a_B, s_B = self._run_refinement(t_p, t_a, best_n_B, best_h_B)

        # --- C: Multi-Start Hertzian ---
        best_loss_C, best_h_C = float('inf'), None
        for _ in range(n_starts):
            n_hertz = torch.ones(1, self.n_asp).to(self.device) * 2.0
            h_rand = torch.sort(torch.rand(1, self.n_asp).to(self.device) * self.gen.max_d, dim=1)[0]
            _, h_prb, p_prb, a_prb, _ = self._run_refinement(t_p, t_a, n_hertz, h_rand, lock_n=True)
            if (l := compute_loss(p_prb, a_prb).item()) < best_loss_C:
                best_loss_C, best_h_C = l, h_prb
        n_C, h_C, p_C, a_C, s_C = self._run_refinement(t_p, t_a, torch.ones_like(n_hertz)*2.0, best_h_C, lock_n=True)

        # Return detached numpy arrays ready for matplotlib
        return {
            "Target": (t_p[0].cpu().numpy(), t_a[0].cpu().numpy(), t_s[0].cpu().numpy()),
            "Surrogate": (p_A[0].cpu().numpy(), a_A[0].cpu().numpy(), s_A[0].cpu().numpy()),
            "MS_General": (p_B[0].cpu().numpy(), a_B[0].cpu().numpy(), s_B[0].cpu().numpy()),
            "Hertz": (p_C[0].cpu().numpy(), a_C[0].cpu().numpy(), s_C[0].cpu().numpy())
        }

    def prepare_nn_input(self, native_p, native_alpha, native_s):
        """Converts raw physics outputs into the normalized hat arrays & scalars for the new MLP."""
        # 1. Extract absolute maximums
        p_max = torch.clamp(native_p[:, -1:], min=1e-12)
        a_max = torch.clamp(native_alpha[:, -1:], min=1e-12)

        # 2. Normalize
        p_hat = native_p / p_max
        a_hat = native_alpha / a_max
        s_hat = native_s * (a_max / p_max)

        # 3. Interpolate onto standard [0,1] grid
        a_interp = batched_interp1d(
            self.p_hat_grid, p_hat, a_hat, pad_value=1.0)
        s_interp = batched_interp1d(
            self.p_hat_grid, p_hat, s_hat, pad_value=0.0)

        # 4. Pack for model
        x_arrays = torch.stack([a_interp, s_interp], dim=1)
        x_scalars_log = torch.log10(torch.cat([p_max, a_max], dim=1) + 1e-12)

        return x_arrays, x_scalars_log

    # -------------------------------------------------------------------------
    # MAIN VALIDATION LOGIC
    # -------------------------------------------------------------------------

    def get_test_set_indices_by_category(self):
        print("[Validator] Reconstructing Test Split to find unseen samples...")
        total_len = self.gen.total_samples
        train_len, val_len = int(0.8 * total_len), int(0.1 * total_len)

        _, _, test_ds = random_split(range(total_len), [
                                     train_len, val_len, total_len - train_len - val_len], generator=torch.Generator().manual_seed(42))

        categorized = {k: [] for k in self.gen.ranges}
        for idx in test_ds.indices:
            for cat, (start, end) in self.gen.ranges.items():
                if start <= idx < end:
                    categorized[cat].append(idx)
                    break
        return categorized

    def validate_on_test_set(self, refine=True):
        for category, indices in self.get_test_set_indices_by_category().items():
            if not indices:
                continue

            idx = np.random.choice(indices) + 1
            print(f"\nValidating {category.upper()} on Test Sample #{idx}...")

            t_p, t_a, t_s, gt_n, gt_h, _ = self.gen.get_custom_sample(
                idx, category)
            x_arr, x_scal_log = self.prepare_nn_input(t_p, t_a, t_s)

            with torch.no_grad():
                n_pred, h_pred = self.model(x_arr, x_scal_log)
                h_anchored = torch.sort(h_pred, dim=1)[
                    0] - torch.sort(h_pred, dim=1)[0][:, 0:1]
                p_nn, a_nn, s_nn = self.phys(
                    h_anchored, n_pred, self.gen.t_w, self.gen.indentations, k_steepness=1e5)

            if refine:
                n_ref, h_ref, p_ref, a_ref, s_ref = self._run_refinement(
                    t_p, t_a, n_pred, h_pred)
                self.plot_triple_comparison(
                    t_p, t_a, t_s, gt_n, gt_h,
                    p_nn, a_nn, s_nn, n_pred, h_anchored,
                    p_ref, a_ref, s_ref, n_ref, h_ref, title=f"Test: {category} (#{idx})"
                )
            else:
                self.plot_comparison(
                    t_p, t_a, t_s, gt_n, gt_h,
                    p_nn, a_nn, s_nn, n_pred, h_anchored, title=f"Test: {category} (#{idx})"
                )

    def validate_designed(self, target_type="linear", refine=False):
        print(
            f"\n[Validator] Generating fresh synthetic target: {target_type}...")
        t_p, t_a, t_s, title = self._get_target(target_type)
        x_arr, x_scal_log = self.prepare_nn_input(t_p, t_a, t_s)

        with torch.no_grad():
            n_pred, h_pred = self.model(x_arr, x_scal_log)
            h_anchored = torch.sort(h_pred, dim=1)[
                0] - torch.sort(h_pred, dim=1)[0][:, 0:1]
            p_nn, a_nn, s_nn = self.phys(
                h_anchored, n_pred, self.gen.t_w, self.gen.indentations, k_steepness=1e5)

        if refine:
            n_ref, h_ref, p_ref, a_ref, s_ref = self._run_refinement(
                t_p, t_a, n_pred, h_pred)
            self.plot_triple_comparison(
                t_p, t_a, t_s, None, None, p_nn, a_nn, s_nn, n_pred, h_anchored,
                p_ref, a_ref, s_ref, n_ref, h_ref, title=f"Refined: {title}"
            )
        else:
            self.plot_comparison(
                t_p, t_a, t_s, None, None, p_nn, a_nn, s_nn, n_pred, h_anchored, title=f"Unseen: {title}"
            )

    def generate_ms_hertz_summary(self, n_starts=50):
        print("\n=== Generating 6-Panel Publication Summary ===")
        
        targets = ["saturate", "bilinear", "linear"]
        titles = ["Saturating", "Bilinear", "Linear"]
        
        fig, axs = plt.subplots(2, 3, figsize=(9.8, 5.6), sharex='col')
        C_GT, C_CNN, C_GEN, C_HERTZ = '#333333', '#0072B2', '#009E73', '#D55E00'

        for col, (target_type, title) in enumerate(zip(targets, titles)):
            # 1. Get the data
            res = self._evaluate_baseline(target_type, n_starts)
            
            t_p, t_a, t_s = res["Target"]
            p_A, a_A, s_A = res["Surrogate"]
            p_B, a_B, s_B = res["MS_General"]
            p_C, a_C, s_C = res["Hertz"]

            # 2. Top Row: Area vs Load
            ax_a = axs[0, col]
            ax_a.plot(t_p, t_a, color=C_GT, lw=3, label="Target")
            ax_a.plot(p_C, a_C, color=C_HERTZ, ls='-', lw=2, label="Hertz (n=2)", alpha=0.8)   
            ax_a.plot(p_B, a_B, color=C_GEN, ls='-', lw=2, label="Multistart", alpha=0.8)
            ax_a.plot(p_A, a_A, color=C_CNN, ls='-', lw=2, label="Surrogate", alpha=0.8)
            
            ax_a.set_title(title, fontsize=14, fontweight='bold')
            if col == 0:
                ax_a.set_ylabel(r"$\alpha$", fontsize=12)
            
            # 3. Bottom Row: Stiffness vs Load
            ax_s = axs[1, col]
            ax_s.plot(t_p, t_s, color=C_GT, lw=3)
            ax_s.plot(p_C, s_C, color=C_HERTZ, ls='-', lw=2, alpha=0.8)
            ax_s.plot(p_B, s_B, color=C_GEN, ls='-', lw=2, alpha=0.8)
            ax_s.plot(p_A, s_A, color=C_CNN, ls='-', lw=2, alpha=0.8)
            
            ax_s.set_xlabel("$P^*$", fontsize=12)
            if col == 0:
                ax_s.set_ylabel("$S^*$", fontsize=12) # Fixed the missing asterisk here too

            # 4. Styling
            for ax in [ax_a, ax_s]:
                ax.grid(True, alpha=0.2)
                ax.tick_params(direction='in', top=True, right=True)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # Add a single legend to the first plot
        axs[0, 0].legend(loc="lower right", fontsize=12)
        
        # --- Add Standard Publication Labels (a)-(f) ---
        for i, ax in enumerate(axs.flat):
            ax.text(0.03, 0.95, f"({chr(97+i)})", transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left')

        # h_pad reduces the vertical gap to visually group the shared axes closer together
        plt.tight_layout(h_pad=0.5) 
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/ms_hertz_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()


    def validate_optimization_baseline(self, target_type="quadratic", n_starts=50):
        print(
            f"\n[Baseline] Tri-Comparison for {target_type} (CNN vs General Multi vs Hertz Multi)...")

        t_p, t_a, t_s, title = self._get_target(target_type)
        x_arr, x_scal_log = self.prepare_nn_input(t_p, t_a, t_s)

        def compute_loss(pred_p, pred_a):
            # Compute physical MSE by interpolating predictions onto the exact target P points
            pred_a_aligned = batched_interp1d(
                t_p[0], pred_p, pred_a, pad_value=pred_a[0, -1].item())
            return torch.mean((pred_a_aligned - t_a)**2)

        # --- A: MLP Surrogate ---
        with torch.no_grad():
            n_cnn, h_cnn = self.model(x_arr, x_scal_log)
        n_A, h_A, p_A, a_A, s_A = self._run_refinement(t_p, t_a, n_cnn, h_cnn)
        loss_A = compute_loss(p_A, a_A).item()
        print(f"  > [A] Surrogate (n=var): {loss_A:.6e}")

        # --- B: Multi-Start General ---
        best_loss_B, best_n_B, best_h_B = float('inf'), None, None
        gamma_range = self.gamma_max - self.gamma_min
        
        for _ in range(n_starts):
            n_rand = self.gamma_min + torch.rand(1, self.n_asp).to(self.device) * gamma_range
            h_rand = torch.sort(torch.rand(1, self.n_asp).to(self.device) * self.gen.max_d, dim=1)[0]
            
            n_prb, h_prb, p_prb, a_prb, _ = self._run_refinement(t_p, t_a, n_rand, h_rand)
            if (l := compute_loss(p_prb, a_prb).item()) < best_loss_B:
                best_loss_B, best_n_B, best_h_B = l, n_prb, h_prb
        n_B, h_B, p_B, a_B, s_B = self._run_refinement(
            t_p, t_a, best_n_B, best_h_B)
        loss_B = compute_loss(p_B, a_B).item()
        print(f"  > [B] MS General (n=var): {loss_B:.6e}")

        # --- C: Multi-Start Hertzian ---
        best_loss_C, best_h_C = float('inf'), None
        for _ in range(n_starts):
            n_hertz = torch.ones(1, self.n_asp).to(self.device) * 2.0
            h_rand = torch.sort(torch.rand(1, self.n_asp).to(
                self.device) * self.gen.max_d, dim=1)[0]
            _, h_prb, p_prb, a_prb, _ = self._run_refinement(
                t_p, t_a, n_hertz, h_rand, lock_n=True)
            if (l := compute_loss(p_prb, a_prb).item()) < best_loss_C:
                best_loss_C, best_h_C = l, h_prb
        n_C, h_C, p_C, a_C, s_C = self._run_refinement(
            t_p, t_a, torch.ones_like(n_hertz)*2.0, best_h_C, lock_n=True)
        loss_C = compute_loss(p_C, a_C).item()
        print(f"  > [C] MS Hertz (n=2):   {loss_C:.6e}")

        # --- PLOTTING DIRECTLY ---
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        C_GT, C_CNN, C_GEN, C_HERTZ = '#333333', '#0072B2', '#009E73', '#D55E00'

        def style_ax(ax):
            ax.grid(True, alpha=0.15)
            ax.tick_params(direction='in', top=True, right=True)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        def plot_stem(ax, x, y, color, marker, label):
            markerline, stemlines, _ = ax.stem(x, y, basefmt=" ")
            plt.setp(markerline, color=color, marker=marker,
                     markersize=6, label=label)
            plt.setp(stemlines, color=color, lw=1.5, alpha=0.5)

        # Plot absolute arrays directly
        t_p_np, t_a_np, t_s_np = t_p[0].cpu().numpy(
        ), t_a[0].cpu().numpy(), t_s[0].cpu().numpy()

        axs[0, 0].plot(t_p_np, t_a_np, color=C_GT, lw=3, label="Target")
        axs[0, 0].plot(p_C[0].cpu().numpy(), a_C[0].cpu().numpy(),
                       color=C_HERTZ, ls='-.', lw=2, label=f"Hertz (n=2)")
        axs[0, 0].plot(p_B[0].cpu().numpy(), a_B[0].cpu().numpy(),
                       color=C_GEN, ls='--', lw=2, label=f"MS General (n=var)")
        axs[0, 0].plot(p_A[0].cpu().numpy(), a_A[0].cpu().numpy(),
                       color=C_CNN, ls=':', lw=4, label="Surrogate")
        axs[0, 0].set(title="Baseline: Area vs Load",
                      ylabel="Contact Fraction α")
        style_ax(axs[0, 0])
        axs[0, 0].legend()

        axs[1, 0].plot(t_p_np, t_s_np, color=C_GT, lw=3)
        axs[1, 0].plot(p_C[0].cpu().numpy(), s_C[0].cpu().numpy(),
                       color=C_HERTZ, ls='-.', lw=2)
        axs[1, 0].plot(p_B[0].cpu().numpy(), s_B[0].cpu().numpy(),
                       color=C_GEN, ls='--', lw=2)
        axs[1, 0].plot(p_A[0].cpu().numpy(), s_A[0].cpu().numpy(),
                       color=C_CNN, ls=':', lw=4)
        axs[1, 0].set(title="Baseline: Stiffness vs Load",
                      xlabel="Nominal Pressure P*", ylabel="Stiffness S*")
        style_ax(axs[1, 0])

        idx = np.arange(self.n_asp)
        s_A, s_B, s_C = [torch.argsort(h[0]).cpu().numpy()
                         for h in [h_A, h_B, h_C]]

        plot_stem(axs[0, 1], idx - 0.2, h_C[0]
                  [s_C].cpu().numpy(), C_HERTZ, 'D', 'Hertz')
        plot_stem(axs[0, 1], idx, h_B[0][s_B].cpu().numpy(),
                  C_GEN, '^', 'MS General')
        plot_stem(axs[0, 1], idx + 0.2, h_A[0]
                  [s_A].cpu().numpy(), C_CNN, 's', 'Surrogate')
        axs[0, 1].set(title="Optimized Heights", ylabel="h [m]")
        axs[0, 1].legend()

        buffer = (self.gamma_max - self.gamma_min) * 0.1
        y_bottom = self.gamma_min - buffer
        y_top = self.gamma_max + buffer

        axs[1, 1].axhline(2.0, color=C_HERTZ, ls='-', alpha=0.3)
        plot_stem(axs[1, 1], idx - 0.2, n_C[0][s_C].detach().cpu().numpy(), C_HERTZ, 'D', 'Hertz')
        plot_stem(axs[1, 1], idx, n_B[0][s_B].detach().cpu().numpy(), C_GEN, '^', 'MS General')
        plot_stem(axs[1, 1], idx + 0.2, n_A[0][s_A].detach().cpu().numpy(), C_CNN, 's', 'Surrogate')
        
        axs[1, 1].set(title="Optimized Exponents",
                      xlabel="Asperity Index", ylabel="γ [-]", ylim=(y_bottom, y_top))
        axs[1, 1].legend(loc='lower right', fontsize='small')

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/baseline_3way_{target_type}.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------------
    # PLOTTING LOGIC
    # -------------------------------------------------------------------------
    def plot_test_set_overview(self, save_path="plots/fig2_test_set_overview.png"):
        print("\n[Validator] Generating Dataset Overview Figure from Test Set...")
        categorized_indices = self.get_test_set_indices_by_category()

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs = axs.flatten()
        C_DATA = '#2F4F4F'
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

        for i, (category, indices) in enumerate(categorized_indices.items()):
            if i >= 6:
                break
            ax = axs[i]

            if not indices:
                ax.text(
                    0.5, 0.5, f"No Test Data:\n{category}", ha='center', va='center')
                continue

            idx = np.random.choice(indices) + 1
            t_p, t_a, _, _, _, _ = self.gen.get_custom_sample(idx, category)

            # Plot the raw unscaled physics directly!
            t_p_np = t_p[0].cpu().numpy()
            t_a_np = t_a[0].cpu().numpy()

            ax.plot(t_p_np, t_a_np, color=C_DATA,
                    linewidth=3, solid_capstyle='round')

            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_title(category.replace("_", " ").title(),
                         fontsize=14, pad=12, fontweight='bold')
            ax.text(0.05, 0.90, panel_labels[i], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')

            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax.set_xlim(0, t_p_np.max() * 1.05)  # Local scaling
            ax.set_ylim(0, t_a_np.max() * 1.1)

            ax.set_xlabel('$P^*$', fontsize=13)
            ax.set_ylabel(r'$\alpha$', fontsize=13)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25, wspace=0.2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

    def plot_test_set_reconstructions_grid(self, save_path="plots/dataset_nn_reconstructions.pdf"):
        """
        Generates a 2x3 grid showing one unseen test sample per category,
        overlaying the Target physics curve with the zero-shot NN Prediction.
        """
        print("\n[Validator] Generating 6-Panel NN Reconstruction Grid...")
        categorized_indices = self.get_test_set_indices_by_category()

        # Setup 2x3 grid (better for landscape paper layouts)
        fig, axs = plt.subplots(2, 3, figsize=(
            10, 6), sharex=True, sharey=True)
        axs = axs.flatten()

        # Styling palette
        C_GT = '#333333'  # Dark Charcoal for Target
        C_NN = '#0072B2'  # Teal/Blue for NN Prediction
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

        paper_names = {
            "linear": "Linear Coulomb",
            "bilinear": "Bilinear Transition",
            "saturating": "Saturating",
            "bimodal": "Bimodal",
            "sparse": "Stratified",
            "lhs": "LHS",
            "random_sum": "Mixed",
            "wall": "Coplanar",
            "exiled": "Truncated"
        }

        for i, (category, indices) in enumerate(categorized_indices.items()):
            if i >= 6:
                break  # Safety limit for 6 subplots

            display_title = paper_names.get(
                category, category.replace("_", " ").title())

            ax = axs[i]
            if not indices:
                ax.text(
                    0.5, 0.5, f"No Test Data:\n{category}", ha='center', va='center')
                continue

            # 1. Pick random unseen sample
            idx = np.random.choice(indices) + 1

            # 2. Fetch Ground Truth
            t_p, t_a, t_s, _, _, _ = self.gen.get_custom_sample(idx, category)

            # rename categories

            # 3. Prepare Input and Run Model
            x_arr, x_scal_log = self.prepare_nn_input(t_p, t_a, t_s)
            with torch.no_grad():
                n_pred, h_pred = self.model(x_arr, x_scal_log)
                # Anchor heights to 0
                h_anchored = torch.sort(h_pred, dim=1)[
                    0] - torch.sort(h_pred, dim=1)[0][:, 0:1]
                # Run through Sneddon physics engine
                p_nn, a_nn, _ = self.phys(
                    h_anchored, n_pred, self.gen.t_w, self.gen.indentations, k_steepness=1e5)

            # 4. Convert to absolute numpy arrays
            t_p_np = t_p[0].cpu().numpy()
            t_a_np = t_a[0].cpu().numpy()
            p_p_np = p_nn[0].cpu().numpy()
            p_a_np = a_nn[0].cpu().numpy()

            # 5. Plotting
            ax.plot(t_p_np, t_a_np, color=C_GT, lw=3,
                    label="Target" if i == 0 else "")
            ax.plot(p_p_np, p_a_np, color=C_NN, linestyle='--',
                    lw=2.5, label="Prediction" if i == 0 else "")

            # 6. Clean Formatting
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.text(0.5, 0.15, display_title,
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight='bold',
                    ha='center',
                    va='top',
                    # Optional: Add a subtle semi-transparent white box behind the text
                    # so the grid or data lines don't make it hard to read.
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

            ax.text(0.05, 0.90, panel_labels[i], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')

            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

            # Dynamic Local Axis Scaling
            max_p = max(t_p_np.max(), p_p_np.max())
            max_a = max(t_a_np.max(), p_a_np.max())
            # ax.set_xlim(0, max_p * 1.05)
            # ax.set_ylim(0, max_a * 1.1)

            if i >= 3:
                ax.set_xlabel('$P^*$', fontsize=13)
            if i % 3 == 0:
                ax.set_ylabel(r'$\alpha$', fontsize=13)

        # Add global legend to the first panel
        axs[0].legend(loc='best', frameon=False, fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.25)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        print(f"  > Saved 6-panel grid to {save_path}")
        plt.close()

    def plot_comparison(self, t_p, t_a, t_s, gt_n, gt_h, p_nn, a_nn, s_nn, n_nn, h_nn, title):
        fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex='col')
        C_GT, C_NN = '#333333', '#0072B2'

        def style_ax(ax):
            ax.grid(True, alpha=0.15, color='gray')
            ax.tick_params(axis='both', direction='in', top=True, right=True)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        def plot_stem(ax, x, y, color, marker, label):
            markerline, stemlines, baseline = ax.stem(x, y, basefmt=" ")
            plt.setp(markerline, color=color, marker=marker, markersize=7)
            plt.setp(stemlines, color=color,
                     linestyle='-', linewidth=2, alpha=0.6)
            markerline.set_label(label)

        # Plot raw absolute data
        t_p_np, t_a_np, t_s_np = t_p[0].cpu().numpy(), t_a[0].cpu().numpy(), t_s[0].cpu().numpy()
        p_p_np, p_a_np, p_s_np = p_nn[0].cpu().numpy(), a_nn[0].cpu().numpy(), s_nn[0].cpu().numpy()

        axs[0, 0].plot(t_p_np, t_a_np, color=C_GT, lw=3, label="Target")
        axs[0, 0].plot(p_p_np, p_a_np, color=C_NN, linestyle='--', lw=2.5, label="Prediction")
        axs[0, 0].set(ylabel=r"$\alpha$")
        style_ax(axs[0, 0])
        axs[0, 0].legend()

        axs[1, 0].plot(t_p_np, t_s_np, color=C_GT, lw=3, label="Target")
        axs[1, 0].plot(p_p_np, p_s_np, color=C_NN, linestyle='--', lw=2.5, label="Prediction")
        axs[1, 0].set(xlabel="$P^*$", ylabel="$S^*$")
        style_ax(axs[1, 0])

        idx = np.arange(h_nn.shape[1])
        s_idx_nn = torch.argsort(h_nn[0]).cpu().numpy()

        # Normalizing heights by max_d
        h_norm_nn = h_nn[0][s_idx_nn].detach().cpu().numpy() / self.gen.max_d

        if gt_h is not None:
            s_idx_gt = torch.argsort(gt_h[0]).cpu().numpy()
            h_norm_gt = gt_h[0][s_idx_gt].cpu().numpy() / self.gen.max_d
            
            plot_stem(axs[0, 1], idx - 0.1, h_norm_gt, C_GT, 'o', 'Ground Truth')
            plot_stem(axs[0, 1], idx + 0.1, h_norm_nn, C_NN, 's', 'Prediction')
            plot_stem(axs[1, 1], idx - 0.1, gt_n[0][s_idx_gt].cpu().numpy(), C_GT, 'o', 'Ground Truth')
            plot_stem(axs[1, 1], idx + 0.1, n_nn[0][s_idx_nn].detach().cpu().numpy(), C_NN, 's', 'Prediction')
        else:
            plot_stem(axs[0, 1], idx, h_norm_nn, C_NN, 's', 'Prediction')
            plot_stem(axs[1, 1], idx, n_nn[0][s_idx_nn].detach().cpu().numpy(), C_NN, 's', 'Prediction')

        # Set Dynamic Y-Limits based on config boundaries
        buffer = (self.gamma_max - self.gamma_min) * 0.1
        y_bottom = self.gamma_min - buffer
        y_top = self.gamma_max + buffer

        axs[0, 1].set(ylabel=r"$h / \Delta_{max}$") # xlabel="Asperity Index",
        axs[1, 1].set(xlabel="Asperity Index", ylabel=r"Shape $\gamma$", ylim=(y_bottom, y_top))
        axs[0, 1].grid(True, alpha=0.15)
        axs[1, 1].grid(True, alpha=0.15)

        # Add panel labels
        for i, ax in enumerate(axs.flat):
            ax.text(0.05, 0.95, f"({chr(97+i)})", transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left')
            
        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else title.split(" ")[0].lower()
        plt.tight_layout()
        plt.savefig(f"plots/val_{sname}.png", dpi=150)
        plt.close()


    def plot_triple_comparison(self, t_p, t_a, t_s, gt_n, gt_h, p_nn, a_nn, s_nn, n_nn, h_nn, p_ref, a_ref, s_ref, n_ref, h_ref, title):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        C_GT, C_NN, C_REF = '#333333', '#0072B2', '#D55E00'

        def style_ax(ax):
            ax.grid(True, alpha=0.15, color='gray')
            ax.tick_params(axis='both', direction='in', top=True, right=True)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        def plot_stem(ax, x, y, color, marker, label):
            markerline, stemlines, baseline = ax.stem(x, y, basefmt=" ")
            plt.setp(markerline, color=color, marker=marker, markersize=7)
            plt.setp(stemlines, color=color, linestyle='-', linewidth=2, alpha=0.6)
            markerline.set_label(label)

        t_p_np, t_a_np, t_s_np = t_p[0].cpu().numpy(), t_a[0].cpu().numpy(), t_s[0].cpu().numpy()
        p_p_np, p_a_np, p_s_np = p_nn[0].cpu().numpy(), a_nn[0].cpu().numpy(), s_nn[0].cpu().numpy()
        r_p_np, r_a_np, r_s_np = p_ref[0].cpu().numpy(), a_ref[0].cpu().numpy(), s_ref[0].cpu().numpy()

        axs[0, 0].plot(t_p_np, t_a_np, color=C_GT, lw=3, label="Target")
        axs[0, 0].plot(p_p_np, p_a_np, color=C_NN, linestyle='--', lw=2.5, label="Zero-Shot (NN)")
        axs[0, 0].plot(r_p_np, r_a_np, color=C_REF, linestyle=':', lw=3.5, label="Refined (Opt)")
        axs[0, 0].set(title="Contact Area vs Load", xlabel="Nominal Pressure P*", ylabel="Contact Fraction α")
        style_ax(axs[0, 0])
        axs[0, 0].legend()

        axs[1, 0].plot(t_p_np, t_s_np, color=C_GT, lw=3, label="Target")
        axs[1, 0].plot(p_p_np, p_s_np, color=C_NN, linestyle='--', lw=2.5, label="Zero-Shot (NN)")
        axs[1, 0].plot(r_p_np, r_s_np, color=C_REF, linestyle=':', lw=3.5, label="Refined (Opt)")
        axs[1, 0].set(title="Topological Stiffness", xlabel="Nominal Pressure P*", ylabel="Stiffness S*")
        style_ax(axs[1, 0])
        axs[1, 0].legend()

        idx = np.arange(h_nn.shape[1])
        s_nn_idx = torch.argsort(h_nn[0]).cpu().numpy()
        s_ref_idx = torch.argsort(h_ref[0]).cpu().numpy()

        # Normalizing heights by max_d
        h_norm_nn = h_nn[0][s_nn_idx].detach().cpu().numpy() / self.gen.max_d
        h_norm_ref = h_ref[0][s_ref_idx].detach().cpu().numpy() / self.gen.max_d

        if gt_h is not None:
            s_gt_idx = torch.argsort(gt_h[0]).cpu().numpy()
            h_norm_gt = gt_h[0][s_gt_idx].cpu().numpy() / self.gen.max_d
            
            plot_stem(axs[0, 1], idx - 0.15, h_norm_gt, C_GT, 'o', 'Ground Truth')
            plot_stem(axs[0, 1], idx, h_norm_nn, C_NN, 's', 'NN Pred')
            plot_stem(axs[0, 1], idx + 0.15, h_norm_ref, C_REF, '^', 'Refined')

            plot_stem(axs[1, 1], idx - 0.15, gt_n[0][s_gt_idx].cpu().numpy(), C_GT, 'o', 'Ground Truth')
            plot_stem(axs[1, 1], idx, n_nn[0][s_nn_idx].detach().cpu().numpy(), C_NN, 's', 'NN Pred')
            plot_stem(axs[1, 1], idx + 0.15, n_ref[0][s_ref_idx].detach().cpu().numpy(), C_REF, '^', 'Refined')
        else:
            plot_stem(axs[0, 1], idx - 0.1, h_norm_nn, C_NN, 's', 'NN Pred')
            plot_stem(axs[0, 1], idx + 0.1, h_norm_ref, C_REF, '^', 'Refined')

            plot_stem(axs[1, 1], idx - 0.1, n_nn[0][s_nn_idx].detach().cpu().numpy(), C_NN, 's', 'NN Pred')
            plot_stem(axs[1, 1], idx + 0.1, n_ref[0][s_ref_idx].detach().cpu().numpy(), C_REF, '^', 'Refined')

        # Set Dynamic Y-Limits based on config boundaries
        buffer = (self.gamma_max - self.gamma_min) * 0.1
        y_bottom = self.gamma_min - buffer
        y_top = self.gamma_max + buffer

        axs[0, 1].set(title="Predicted Heights", xlabel="Asperity Index", ylabel=r"$h / \Delta_{max}$")
        axs[1, 1].set(title="Shape Exponent Distribution", xlabel="Asperity Index", ylabel=r"Exponent $\gamma$ [-]", ylim=(y_bottom, y_top))
        axs[0, 1].grid(True, alpha=0.15)
        axs[1, 1].grid(True, alpha=0.15)
        axs[0, 1].legend()
        axs[1, 1].legend(loc='lower right', fontsize='small')

        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else "sample"
        plt.tight_layout()
        plt.savefig(f"plots/val_test_{sname}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    val = UnifiedValidator("config.yaml")
    set_seed()

    val.plot_test_set_reconstructions_grid()
    # val.plot_test_set_overview()

    # Optional baseline executions
    # val.validate_on_test_set(refine=False)
    val.validate_designed(target_type="linear", refine=True)
    val.validate_designed(target_type="saturate", refine=True)
    val.validate_designed(target_type="bilinear", refine=True)

    val.generate_ms_hertz_summary(n_starts=50)

    # val.validate_optimization_baseline(target_type="saturate")
    # val.validate_optimization_baseline(target_type="bilinear")
    # val.validate_optimization_baseline(target_type="linear")
    # val.validate_optimization_baseline(target_type="quadratic")
