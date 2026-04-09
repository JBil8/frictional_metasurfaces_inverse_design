import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split

# Ensure pathing works when executed from inside the validation folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from validation.targets import TargetGenerator
except ImportError:
    from targets import TargetGenerator

from utils.seeding import set_seed
from utils.normalization import get_theoretical_limits
from utils.config import load_config
from physics.differentiable import AxisymmetricContactLayer
from ml_models.model_mlp import SurfaceInverseModel
from utils.interpolation import batched_interp1d

class UnifiedValidator:
    def __init__(self, cfg_path="config.yaml"):
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join("..", cfg_path)
        self.cfg = load_config(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Establish Intensive Limits
        limits = get_theoretical_limits(self.cfg, self.device)
        self.MAX_P = limits['max_pressure']
        self.MAX_ALPHA = limits['max_alpha']
        self.MAX_S = limits['max_stiff']

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
        
        # 3. Create the global P* grid used for all CNN evaluations
        self.steps = self.cfg['data']['n_steps']
        self.p_star_grid = torch.linspace(0, self.MAX_P, self.steps).to(self.device)

    def prepare_nn_input(self, native_p, native_alpha, native_s):
        """Standardizes a native displacement-based curve for the P*-domain CNN."""
        # Force inputs to be (1, Steps) to ensure they are 2D for the interp tool
        p_2d = native_p.view(1, -1)
        a_2d = native_alpha.view(1, -1)
        s_2d = native_s.view(1, -1)

        # 1. Align to global P* grid
        # self.p_star_grid is already 1D, so we leave it as is
        aligned_alpha = batched_interp1d(self.p_star_grid, p_2d, a_2d, pad_value=-1.0)
        aligned_s = batched_interp1d(self.p_star_grid, p_2d, s_2d, pad_value=-1.0)

        # 2. Normalize valid regions safely
        norm_alpha = torch.where(aligned_alpha != -1.0, aligned_alpha / self.MAX_ALPHA, -1.0)
        norm_s = torch.where(aligned_s != -1.0, aligned_s / self.MAX_S, -1.0)
        
        # Ensure p_grid is also (1, Steps) for stacking
        norm_p = (self.p_star_grid / self.MAX_P).view(1, -1)

        # 3. Stack into (1, 3, Steps)
        return torch.stack([norm_p, norm_alpha, norm_s], dim=1)

    def get_test_set_indices_by_category(self):
        print("[Validator] Reconstructing Test Split to find unseen samples...")
        total_len = self.gen.total_samples
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        dataset = range(total_len)
        generator = torch.Generator().manual_seed(42)
        _, _, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

        ranges = self.gen.ranges
        categorized_test_indices = {k: [] for k in ranges}

        for idx in test_ds.indices:
            for cat, (start, end) in ranges.items():
                if start <= idx < end:
                    categorized_test_indices[cat].append(idx)
                    break

        return categorized_test_indices

    def validate_on_test_set(self, refine=True):
        test_buckets = self.get_test_set_indices_by_category()

        for category, indices in test_buckets.items():
            if len(indices) == 0: continue
            
            idx = np.random.choice(indices)
            print(f"Validating {category.upper()} on Test Sample #{idx}...")

            t_p, t_alpha, t_s, gt_n, gt_h, title = self.gen.get_custom_sample(idx, category)
            nn_input = self.prepare_nn_input(t_p, t_alpha, t_s)

            with torch.no_grad():
                n_pred, h_pred = self.model(nn_input)
                p_nn, alpha_nn, s_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations, k_steepness=1e8)

            if refine:
                n_ref, h_ref, p_ref, alpha_ref, s_ref = self.refine_prediction(t_alpha, t_p, n_pred, h_pred)
                self.plot_triple_comparison(t_p, t_alpha, t_s, gt_n, gt_h,
                                            p_nn, alpha_nn, s_nn, n_pred, h_pred,
                                            p_ref, alpha_ref, s_ref, n_ref, h_ref,
                                            f"Test: {category} (#{idx})")
            else:
                self.plot_comparison(t_p, t_alpha, t_s, gt_n, gt_h, p_nn, alpha_nn, s_nn, n_pred, h_pred, f"Test: {category} (#{idx})")

    def refine_prediction(self, target_alpha, target_p, n_init, h_init, steps=50):
        print(f"  > Refinement: Optimizing intensive topology in P* domain...")

        n_opt = n_init.clone().detach().requires_grad_(True)
        h_opt = h_init.clone().detach().requires_grad_(True)

        optimizer = optim.LBFGS([n_opt, h_opt], lr=0.5, max_iter=20, line_search_fn='strong_wolfe')
        
        # FIX: Force 2D shape (1, Steps)
        t_p_2d = target_p.view(1, -1)
        t_a_2d = target_alpha.view(1, -1)

        # Target must be aligned to grid once for the MSE loss
        t_alpha_aligned = batched_interp1d(self.p_star_grid, t_p_2d, t_a_2d, pad_value=-1.0)
        target_mask = (t_alpha_aligned != -1.0).float()
        valid_elements = torch.sum(target_mask) + 1e-8

        for i in range(steps):
            def closure():
                optimizer.zero_grad()
                h_sorted, _ = torch.sort(h_opt, dim=1)
                h_sorted = h_sorted - h_sorted[:, 0:1]

                # phys returns (Batch, Steps) - usually (1, 500)
                p_raw, a_raw, _ = self.phys(h_sorted, n_opt, self.gen.t_w, self.gen.indentations, k_steepness=1e8)
                
                # Ensure 2D for interpolation
                p_raw_2d = p_raw.view(1, -1)
                a_raw_2d = a_raw.view(1, -1)

                a_aligned = batched_interp1d(self.p_star_grid, p_raw_2d, a_raw_2d, pad_value=-1.0)
                
                # Masked Loss
                squared_err = torch.pow(a_aligned - t_alpha_aligned, 2)
                loss = torch.sum(squared_err * target_mask) / valid_elements
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
            except Exception as e:
                print(f"    [Warning] Optimization step failed: {e}")
                break

        with torch.no_grad():
            h_final, _ = torch.sort(h_opt, dim=1)
            h_final = h_final - h_final[:, 0:1]
            p_final, alpha_final, s_final = self.phys(h_final, n_opt, self.gen.t_w, self.gen.indentations, k_steepness=1e8)

        return n_opt, h_final, p_final, alpha_final, s_final

    def validate_designed(self, target_type="linear", refine=False):
        print(f"[Validator] Generating fresh synthetic target: {target_type}...")

        if target_type == "linear":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_linear_coulomb()
        elif target_type == "saturate":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_saturating()
        elif target_type == "bilinear":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_bilinear()
        else:
            return

        nn_input = self.prepare_nn_input(t_p, t_alpha, t_s)

        with torch.no_grad():
            n_pred, h_pred = self.model(nn_input)
            p_nn, alpha_nn, s_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations, k_steepness=1e8)

        if refine:
            n_ref, h_ref, p_ref, alpha_ref, s_ref = self.refine_prediction(t_alpha, t_p, n_pred, h_pred)
            self.plot_triple_comparison(t_p, t_alpha, t_s, None, None, p_nn, alpha_nn, s_nn, n_pred, h_pred, p_ref, alpha_ref, s_ref, n_ref, h_ref, f"Refined: {title}")
        else:
            self.plot_comparison(t_p, t_alpha, t_s, None, None, p_nn, alpha_nn, s_nn, n_pred, h_pred, f"Unseen: {title}")

    def validate_optimization_baseline(self, target_type="bilinear", n_starts=10):
        print(f"[Baseline] Comparing CNN vs Multi-Start ({n_starts} guesses) for {target_type}...")

        if target_type == "bilinear":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_bilinear()
        elif target_type == "saturate":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_saturating()
        elif target_type == "linear":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_linear_coulomb()
        else:
            return

        nn_input = self.prepare_nn_input(t_p, t_alpha, t_s)

        print("  > Strategy A: CNN Initialization...")
        with torch.no_grad():
            n_cnn, h_cnn = self.model(nn_input)
        
        n_A, h_A, p_A, alpha_A, s_A = self.refine_prediction(t_alpha, t_p, n_cnn, h_cnn, steps=100)
        
        def compute_p_star_loss(pred_p, pred_alpha):
            # Flatten both to (1, Steps)
            t_p_2d = t_p.view(1, -1)
            t_a_2d = t_alpha.view(1, -1)
            p_p_2d = pred_p.view(1, -1)
            p_a_2d = pred_alpha.view(1, -1)

            t_aligned = batched_interp1d(self.p_star_grid, t_p_2d, t_a_2d, pad_value=-1.0)
            p_aligned = batched_interp1d(self.p_star_grid, p_p_2d, p_a_2d, pad_value=-1.0)
            mask = (t_aligned != -1.0).float()
            return torch.sum(torch.pow(p_aligned - t_aligned, 2) * mask) / (torch.sum(mask) + 1e-8)

        loss_A = compute_p_star_loss(p_A, alpha_A)
        print(f"    [CNN] Final Alpha Error: {loss_A.item():.6f}")

        print(f"  > Strategy B: Multi-Start ({n_starts} random guesses)...")
        best_loss_B = float('inf')
        best_n_B = None
        best_h_B = None

        for k in range(n_starts):
            n_rand = torch.rand(1, self.cfg['physics']['n_asperities']).to(self.device) * 2.0 + 1.0
            h_rand = torch.rand(1, self.cfg['physics']['n_asperities']).to(self.device) * self.gen.max_d
            h_rand, _ = torch.sort(h_rand, dim=1)
            h_rand = h_rand - h_rand[:, 0:1]

            n_probe, h_probe, p_probe, alpha_probe, s_probe = self.refine_prediction(
                t_alpha, t_p, n_rand, h_rand, steps=20)

            probe_loss = compute_p_star_loss(p_probe, alpha_probe)

            if probe_loss < best_loss_B:
                best_loss_B = probe_loss
                best_n_B = n_probe
                best_h_B = h_probe

        print(f"  > Refining Best Random Candidate...")
        n_B, h_B, p_B, alpha_B, s_B = self.refine_prediction(t_alpha, t_p, best_n_B, best_h_B, steps=80)
        loss_B = compute_p_star_loss(p_B, alpha_B)
        print(f"    [Multi-Start] Final Alpha Error: {loss_B.item():.6f}")

    def plot_comparison(self, t_p, t_a, t_s, gt_n, gt_h, p_nn, a_nn, s_nn, n_pred, h_pred, title):
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        def get_plot_ready(p, val):
            # Force inputs to be exactly 2D: (1, Steps)
            p_2d = p.view(1, -1)
            val_2d = val.view(1, -1)
            aligned = batched_interp1d(self.p_star_grid, p_2d, val_2d, pad_value=-1.0)
            data = aligned[0].cpu().numpy()
            data[data == -1.0] = np.nan
            return data

        p_axis = self.p_star_grid.cpu().numpy()
        max_p_val = t_p.max().item()

        axs[0].plot(p_axis, get_plot_ready(t_p, t_a), 'k-', lw=3, label="Target")
        axs[0].plot(p_axis, get_plot_ready(p_nn, a_nn), 'b--', lw=2, label="Prediction")
        axs[0].set_xlim(0, max_p_val * 1.1)
        axs[0].set_title(f"Contact Area vs Load (α vs P*)")
        axs[0].set_xlabel("Nominal Pressure P*")
        axs[0].set_ylabel("Contact Fraction α")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        width = 0.35
        sorted_idx = torch.argsort(h_pred[0]).cpu().numpy()
        nn_h_sorted = h_pred[0][sorted_idx].detach().cpu().numpy()
        indices = np.arange(len(nn_h_sorted))

        if gt_h is not None:
            gt_h_sorted = gt_h[0][torch.argsort(gt_h[0])].cpu().numpy()
            axs[1].bar(indices - width/2, gt_h_sorted, width, label='Ground Truth', color='k', alpha=0.7)
            axs[1].bar(indices + width/2, nn_h_sorted, width, label='Pred', color='b', alpha=0.7)
        else:
            axs[1].bar(indices, nn_h_sorted, width, label='Pred (Inferred)', color='b', alpha=0.7)

        axs[1].set_title("Predicted Topography Structure")
        axs[1].set_xlabel("Asperity Index")
        axs[1].legend()

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else title.split(" ")[0].lower()
        plt.savefig(f"plots/val_{sname}.png", dpi=150)
        plt.close()

    def plot_triple_comparison(self, t_p, t_a, t_s, gt_n, gt_h,
                               p_nn, a_nn, s_nn, n_pred, h_pred,
                               p_ref, a_ref, s_ref, n_ref, h_ref, title):
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        def get_plot_ready(p, val):
            # Force inputs to be exactly 2D: (1, Steps)
            p_2d = p.view(1, -1)
            val_2d = val.view(1, -1)
            aligned = batched_interp1d(self.p_star_grid, p_2d, val_2d, pad_value=-1.0)
            data = aligned[0].cpu().numpy()
            data[data == -1.0] = np.nan
            return data

        p_axis = self.p_star_grid.cpu().numpy()
        max_p_val = t_p.max().item()

        axs[0].plot(p_axis, get_plot_ready(t_p, t_a), 'k-', lw=3, label="Target")
        axs[0].plot(p_axis, get_plot_ready(p_nn, a_nn), 'b--', lw=2, label="Zero-Shot (NN)")
        axs[0].plot(p_axis, get_plot_ready(p_ref, a_ref), 'r:', lw=4, label="Refined (Opt)")
        
        axs[0].set_xlim(0, max_p_val * 1.1)
        axs[0].set_title(f"Contact Area vs Load (α vs P*)")
        axs[0].set_xlabel("Nominal Pressure P*")
        axs[0].set_ylabel("Contact Fraction α")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        width = 0.25
        sorted_idx = torch.argsort(h_pred[0]).cpu().numpy()
        nn_h_sorted = h_pred[0][sorted_idx].detach().cpu().numpy()
        ref_h_sorted = h_ref[0][sorted_idx].detach().cpu().numpy()
        indices = np.arange(len(nn_h_sorted))

        if gt_h is not None:
            gt_h_sorted = gt_h[0][torch.argsort(gt_h[0])].cpu().numpy()
            axs[1].bar(indices - width, gt_h_sorted, width, label='Ground Truth', color='k', alpha=0.7)
            axs[1].bar(indices, nn_h_sorted, width, label='NN Pred', color='b', alpha=0.7)
            axs[1].bar(indices + width, ref_h_sorted, width, label='Refined', color='r', alpha=0.7)
        else:
            axs[1].bar(indices - width/2, nn_h_sorted, width, label='NN Pred', color='b', alpha=0.7)
            axs[1].bar(indices + width/2, ref_h_sorted, width, label='Refined', color='r', alpha=0.7)

        axs[1].set_title("Predicted Topography Structure")
        axs[1].set_xlabel("Asperity Index")
        axs[1].legend()

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else "sample"
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