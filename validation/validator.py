import sys
import os
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
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, random_split

class UnifiedValidator:
    def __init__(self, cfg_path="config.yaml"):
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join("..", cfg_path)
        self.cfg = load_config(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CRITICAL FIX 1: Intensive limits
        limits = get_theoretical_limits(self.cfg, self.device)
        self.MAX_P = limits['max_pressure']
        self.MAX_ALPHA = limits['max_alpha']
        self.MAX_S = limits['max_stiff']

        # CRITICAL FIX 2: Physics engine takes full config
        self.phys = AxisymmetricContactLayer(cfg=self.cfg).to(self.device)
        self.model = SurfaceInverseModel(self.cfg).to(self.device)

        model_name = self.cfg['model']['name']
        if not os.path.exists(model_name):
            model_name = "../" + model_name
        print(f"[Validator] Loading model from {model_name}...")
        self.model.load_state_dict(torch.load(model_name, map_location=self.device))
        self.model.eval()

        self.gen = TargetGenerator(self.phys, self.cfg, self.device)

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
            if len(indices) == 0:
                continue

            idx = np.random.choice(indices)
            print(f"Validating {category.upper()} on Test Sample #{idx}...")

            # Changed unpack names to Intensive properties
            t_p, t_alpha, t_s, gt_n, gt_h, title = self.gen.get_custom_sample(idx, category)

            nn_input = (t_s / self.MAX_S).unsqueeze(0)

            with torch.no_grad():
                n_pred, h_pred = self.model(nn_input)
                p_nn, alpha_nn, s_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations)

            if refine:
                n_ref, h_ref, p_ref, alpha_ref, s_ref = self.refine_prediction(
                    t_s, n_pred, h_pred, self.gen.indentations)

                self.plot_triple_comparison(t_p, t_s, gt_n, gt_h,
                                            p_nn, s_nn, n_pred, h_pred,
                                            p_ref, s_ref, n_ref, h_ref,
                                            f"Test: {category} (#{idx})")
            else:
                self.plot_comparison(t_p, t_s, gt_n, gt_h, p_nn, s_nn, n_pred, h_pred, f"Test: {category} (#{idx})")

    def refine_prediction(self, target_stiff, n_init, h_init, indent_profile, steps=50):
        print(f"  > Refinement: Optimizing intensive topology (dF/dA)...")

        n_opt = n_init.clone().detach().requires_grad_(True)
        h_opt = h_init.clone().detach().requires_grad_(True)

        optimizer = optim.LBFGS([n_opt, h_opt], lr=0.5, max_iter=20, line_search_fn='strong_wolfe')
        criterion = nn.MSELoss()

        for i in range(steps):
            def closure():
                optimizer.zero_grad()
                h_sorted, _ = torch.sort(h_opt, dim=1)
                h_sorted = h_sorted - h_sorted[:, 0:1]

                _, _, s_pred = self.phys(h_sorted, n_opt, self.gen.t_w, indent_profile)

                loss = criterion(s_pred / self.MAX_S, target_stiff / self.MAX_S)
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
            n_final = n_opt
            p_final, alpha_final, s_final = self.phys(h_final, n_final, self.gen.t_w, indent_profile)

        return n_final, h_final, p_final, alpha_final, s_final

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

        nn_input = (t_s / self.MAX_S).unsqueeze(0)

        print("  > Strategy A: CNN Initialization...")
        with torch.no_grad():
            n_cnn, h_cnn = self.model(nn_input)

        n_A, h_A, p_A, alpha_A, s_A = self.refine_prediction(t_s, n_cnn, h_cnn, self.gen.indentations, steps=100)
        loss_A = torch.nn.functional.mse_loss(s_A / self.MAX_S, t_s / self.MAX_S)
        print(f"    [CNN] Final Stiffness Loss: {loss_A.item():.6f}")

        print(f"  > Strategy B: Multi-Start ({n_starts} random guesses)...")
        best_loss_B = float('inf')
        best_n_B = None
        best_h_B = None

        for k in range(n_starts):
            n_rand = torch.rand(1, self.gen.n_asp).to(self.device) * 7.0 + 1.0
            h_rand = torch.rand(1, self.gen.n_asp).to(self.device) * self.gen.max_d
            h_rand, _ = torch.sort(h_rand, dim=1)
            h_rand = h_rand - h_rand[:, 0:1]

            n_probe, h_probe, p_probe, alpha_probe, s_probe = self.refine_prediction(
                t_s, n_rand, h_rand, self.gen.indentations, steps=20)

            probe_loss = torch.nn.functional.mse_loss(s_probe / self.MAX_S, t_s / self.MAX_S)

            if probe_loss < best_loss_B:
                best_loss_B = probe_loss
                best_n_B = n_probe
                best_h_B = h_probe
                print(f"    [Start #{k+1}] New Best Probe Loss: {probe_loss.item():.6f}")

        print(f"  > Refining Best Random Candidate...")
        n_B, h_B, p_B, alpha_B, s_B = self.refine_prediction(t_s, best_n_B, best_h_B, self.gen.indentations, steps=80)
        loss_B = torch.nn.functional.mse_loss(s_B / self.MAX_S, t_s / self.MAX_S)
        print(f"    [Multi-Start] Final Stiffness Loss: {loss_B.item():.6f}")

        fig = plt.figure(figsize=(10, 6))
        
        ind_np = self.gen.indentations.cpu().numpy().flatten()
        plt.plot(ind_np, t_s.cpu().numpy().flatten(), 'k-', lw=3, label="Target (dF/dA)")
        plt.plot(ind_np, s_A.cpu().detach().numpy().flatten(), 'b--', lw=2, label=f"CNN + Opt (Loss: {loss_A.item():.2e})")
        plt.plot(ind_np, s_B.cpu().detach().numpy().flatten(), 'r:', lw=2, label=f"Multi-Start (Loss: {loss_B.item():.2e})")

        plt.title(f"Topology Optimization Basin: {title}")
        plt.xlabel("Indentation [m]")
        plt.ylabel("Stiffness dF/dA [N/m²]")
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/multistart_{target_type}.png", dpi=150)

    def validate_designed(self, target_type="linear", refine=False):
        print(f"[Validator] Generating fresh synthetic target: {target_type}...")

        if target_type == "linear":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_linear_coulomb()
        elif target_type == "saturate":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_saturating()
        elif target_type == "bilinear":
            t_p, t_alpha, t_s, title = self.gen.get_consistent_bilinear()

        nn_input = (t_s / self.MAX_S).unsqueeze(0)

        with torch.no_grad():
            n_pred, h_pred = self.model(nn_input)
            p_nn, alpha_nn, s_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations)

        if refine:
            n_ref, h_ref, p_ref, alpha_ref, s_ref = self.refine_prediction(t_s, n_pred, h_pred, self.gen.indentations)
            self.plot_triple_comparison(t_p, t_s, None, None, p_nn, s_nn, n_pred, h_pred, p_ref, s_ref, n_ref, h_ref, f"Refined: {title}")
        else:
            self.plot_comparison(t_p, t_s, None, None, p_nn, s_nn, n_pred, h_pred, f"Unseen: {title}")

    def plot_comparison(self, t_p, t_s, gt_n, gt_h, p_nn, s_nn, n_pred, h_pred, title):
        fig = plt.figure(figsize=(14, 6))

        ax1 = plt.subplot(1, 2, 1)
        ind_np = self.gen.indentations.cpu().numpy().flatten()
        ax1.plot(ind_np, t_s.cpu().numpy().flatten(), 'k-', lw=3, label="Target dF/dA")
        ax1.plot(ind_np, s_nn.cpu().numpy().flatten(), 'b--', lw=2, label="Prediction dF/dA")
        ax1.set_title(f"Topology: {title}")
        ax1.set_xlabel("Indentation [m]")
        ax1.set_ylabel("Stiffness [N/m²]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(1, 2, 2)
        width = 0.35
        sorted_idx = torch.argsort(h_pred[0])
        nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
        indices = np.arange(len(nn_h_sorted))

        if gt_h is not None:
            gt_sorted_idx = torch.argsort(gt_h[0])
            gt_h_sorted = gt_h[0][gt_sorted_idx].cpu().numpy()
            ax2.bar(indices - width/2, gt_h_sorted, width, label='Ground Truth', color='black', alpha=0.7)
            ax2.bar(indices + width/2, nn_h_sorted, width, label='Pred', color='blue', alpha=0.7)
        else:
            ax2.bar(indices, nn_h_sorted, width, label='Pred (Inferred)', color='blue', alpha=0.7)

        ax2.set_title("Predicted Topography Structure")
        ax2.set_xlabel("Asperity Index (Sorted)")
        ax2.legend()

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else title.split(" ")[0].lower()
        save_path = f"plots/val_{sname}.png"
        plt.savefig(save_path, dpi=150)
        print(f"[Validator] Saved plot to {save_path}")
        plt.close()

    def plot_triple_comparison(self, t_p, t_s, gt_n, gt_h,
                               p_nn, s_nn, n_pred, h_pred,
                               p_ref, s_ref, n_ref, h_ref, title):
        fig = plt.figure(figsize=(14, 6))

        ax1 = plt.subplot(1, 2, 1)
        ind_np = self.gen.indentations.cpu().numpy().flatten()
        ax1.plot(ind_np, t_s.cpu().numpy().flatten(), 'k-', lw=3, label="Target (GT)")
        ax1.plot(ind_np, s_nn.cpu().numpy().flatten(), 'b--', lw=2, label="Zero-Shot (NN)")
        ax1.plot(ind_np, s_ref.cpu().numpy().flatten(), 'r:', lw=4, label="Refined (Opt)")

        ax1.set_title(f"Topology: {title}")
        ax1.set_xlabel("Indentation [m]")
        ax1.set_ylabel("Stiffness [N/m²]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(1, 2, 2)
        width = 0.25

        if gt_h is not None:
            sorted_idx = torch.argsort(gt_h[0])
            gt_h_sorted = gt_h[0][sorted_idx].cpu().numpy()
            nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
            ref_h_sorted = h_ref[0][sorted_idx].cpu().detach().numpy()
            indices = np.arange(len(gt_h_sorted))

            ax2.bar(indices - width, gt_h_sorted, width, label='Ground Truth', color='black', alpha=0.7)
            ax2.bar(indices, nn_h_sorted, width, label='NN Pred', color='blue', alpha=0.7)
            ax2.bar(indices + width, ref_h_sorted, width, label='Refined', color='red', alpha=0.7)
        else:
            sorted_idx = torch.argsort(h_pred[0])
            nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
            ref_h_sorted = h_ref[0][sorted_idx].cpu().detach().numpy()
            indices = np.arange(len(nn_h_sorted))

            ax2.bar(indices - width/2, nn_h_sorted, width, label='NN Pred', color='blue', alpha=0.7)
            ax2.bar(indices + width/2, ref_h_sorted, width, label='Refined', color='red', alpha=0.7)

        ax2.set_title("Predicted Topography Structure")
        ax2.set_xlabel("Asperity Index")
        ax2.legend()

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else "sample"
        save_path = f"plots/val_test_{sname}.png"
        plt.savefig(save_path, dpi=150)
        print(f"[Validator] Saved plot to {save_path}")
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