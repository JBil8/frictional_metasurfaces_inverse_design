import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from torch.utils.data import TensorDataset, random_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from validation.targets import TargetGenerator
except ImportError:
    from targets import TargetGenerator

from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer
from utils.config import load_config
from utils.normalization import get_theoretical_limits

class UnifiedValidator:
    def __init__(self, cfg_path="config.yaml"):
        if not os.path.exists(cfg_path): cfg_path = os.path.join("..", cfg_path)
        self.cfg = load_config(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Limits & Physics
        limits = get_theoretical_limits(self.cfg, self.device)
        self.MAX_L = limits['max_load']
        self.MAX_A = limits['max_area']
        self.MAX_S = limits['max_stiff']
        
        self.phys = AxisymmetricContactLayer(E_star=self.cfg['physics']['E_star']).to(self.device)
        self.model = SurfaceInverseModel(self.cfg).to(self.device)
        
        model_path = "model_final.pth"
        if not os.path.exists(model_path): model_path = "../model_final.pth"
        print(f"[Validator] Loading model from {model_path}...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.gen = TargetGenerator(self.phys, self.cfg, self.device)

    def get_test_set_indices_by_category(self):
        """
        Reconstructs the Random Split and maps Test Indices back to Categories.
        Returns a dict: {'switch': [idx1, idx2...], 'wall': [...], ...}
        """
        print("[Validator] Reconstructing Test Split to find unseen samples...")
        
        total_len = self.gen.total_samples
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len
        
        # Re-create split
        dataset = range(total_len) 
        generator = torch.Generator().manual_seed(42)
        _, _, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)
        
        ranges = self.gen.ranges 
        
        categorized_test_indices = {k: [] for k in ranges}
        
        # Filter
        for idx in test_ds.indices:
            for cat, (start, end) in ranges.items():
                if start <= idx < end:
                    categorized_test_indices[cat].append(idx)
                    break
                    
        return categorized_test_indices

    def validate_on_test_set(self, refine=True):
        """
        Plots Test Set samples with optional Refinement.
        Visualizes: Target vs Zero-Shot NN vs Refined.
        """
        test_buckets = self.get_test_set_indices_by_category()
        
        for category, indices in test_buckets.items():
            if len(indices) == 0: continue
            
            # Pick random sample
            idx = np.random.choice(indices)
            print(f"Validating {category.upper()} on Test Sample #{idx}...")
            
            # Get Data
            t_l, t_a, gt_n, gt_h, title = self.gen.get_custom_sample(idx, category)
            
            # 1. Zero-Shot Prediction (NN)
            prepend_val = torch.zeros(1, 1).to(self.device)
            raw_stiff = torch.diff(t_l, dim=1, prepend=prepend_val)
            nn_input = torch.cat([t_l/self.MAX_L, t_a/self.MAX_A, raw_stiff/self.MAX_S], dim=0).unsqueeze(0)
            
            with torch.no_grad():
                n_pred, h_pred = self.model(nn_input)
                l_nn, a_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations)
            
            # 2. Refinement (Optional)
            if refine:
                # Use the standard indentation profile for test set samples
                n_ref, h_ref, l_ref, a_ref = self.refine_prediction(t_l, t_a, n_pred, h_pred, self.gen.indentations)
                
                # Plot with THREE curves: Target, NN, Refined
                self.plot_triple_comparison(t_l, t_a, gt_n, gt_h, 
                                          l_nn, a_nn, n_pred, h_pred,
                                          l_ref, a_ref, n_ref, h_ref,
                                          f"Test: {category} (#{idx})")
            else:
                self.plot_comparison(t_l, t_a, gt_n, gt_h, l_nn, a_nn, n_pred, h_pred, f"Test: {category} (#{idx})")

    def refine_prediction(self, target_load, target_area, n_init, h_init, indent_profile, steps=50):
        """
        Refinement using Normalized Loss and explicit Indentation Profile.
        """
        print(f"  > Refinement: Optimizing with correct indentation profile...")
        
        n_opt = n_init.clone().detach().requires_grad_(True)
        h_opt = h_init.clone().detach().requires_grad_(True)
        
        optimizer = optim.LBFGS([n_opt, h_opt], lr=0.5, max_iter=20, line_search_fn='strong_wolfe')
        criterion_mse = nn.MSELoss()
        
        # Normalization factors to balance Load vs Area optimization
        # (Add epsilon to avoid division by zero)
        scale_l = target_load.abs().mean().item() + 1e-6
        scale_a = target_area.abs().mean().item() + 1e-6
        
        for i in range(steps):
            def closure():
                optimizer.zero_grad()
                
                # Sort heights (Topology constraint)
                h_sorted, _ = torch.sort(h_opt, dim=1)
                h_sorted = h_sorted - h_sorted[:, 0:1]
                
                # Soft constraints to keep physics stable without hard clamping
                # We use softplus for heights to keep them positive-ish
                # We simply let n float, but the physics engine handles n < 1 gracefully usually?
                # Actually, let's just stick to the sorted h.
                
                l_pred, a_pred = self.phys(h_sorted, n_opt, self.gen.t_w, indent_profile)
                
                # NORMALIZED LOSS
                loss_l = criterion_mse(l_pred, target_load) / (scale_l**2)
                loss_a = criterion_mse(a_pred, target_area) / (scale_a**2)
                
                loss = loss_l + loss_a
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
            l_final, a_final = self.phys(h_final, n_final, self.gen.t_w, indent_profile)
            
        return n_final, h_final, l_final, a_final

    def validate_designed(self, target_type="linear", refine=False):
        """
        Validates on synthetic targets with AUTOMATIC indentation scaling.
        """
        print(f"[Validator] Generating fresh synthetic target: {target_type}...")
        
        # 1. Generate Target
        if target_type == "linear":
            t_l, t_a, title = self.gen.get_consistent_linear_coulomb()
        elif target_type == "saturate":
            t_l, t_a, title = self.gen.get_consistent_saturating()
        elif target_type == "bilinear":
            t_l, t_a, title = self.gen.get_consistent_bilinear()

        # 2. NN Prediction
        prepend_val = torch.zeros(1, 1).to(self.device)
        raw_stiff = torch.diff(t_l, dim=1, prepend=prepend_val)
        nn_input = torch.cat([t_l/self.MAX_L, t_a/self.MAX_A, raw_stiff/self.MAX_S], dim=0).unsqueeze(0)
        
        with torch.no_grad():
            n_pred, h_pred = self.model(nn_input)
            
            # 3. Dynamic Indentation Check
            # Check stiffness of prediction vs target using standard indent
            l_std, _ = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations)
            current_max = l_std.max().item()
            target_max = t_l.max().item()
            
            # Default to standard indentation
            active_ind = self.gen.indentations
            
            if current_max < target_max:
                print(f"  > Extending indentation (Current: {current_max:.2f}N < Target: {target_max:.2f}N)")
                
                # Calculate required depth ratio
                ratio = target_max / (current_max + 1e-6)
                new_max_d = self.gen.max_d * ratio * 1.1 # +10% buffer
                
                # CRITICAL FIX: Keep the SAME number of steps (500)
                # This ensures the array shapes match for the optimizer!
                active_ind = torch.linspace(0, new_max_d, self.gen.n_steps).unsqueeze(0).to(self.device)
                
                # Re-calculate NN prediction on this new grid
                l_nn, a_nn = self.phys(h_pred, n_pred, self.gen.t_w, active_ind)
            else:
                l_nn, a_nn = l_std, _

        # 4. Refinement
        if refine:
            # Pass the CORRECT indentation profile (active_ind) to the optimizer
            n_ref, h_ref, l_ref, a_ref = self.refine_prediction(t_l, t_a, n_pred, h_pred, active_ind)
            self.plot_triple_comparison(t_l, t_a, None, None, l_nn, a_nn, n_pred, h_pred, l_ref, a_ref, n_ref, h_ref, f"Refined: {title}")
        else:
            self.plot_comparison(t_l, t_a, None, None, l_nn, a_nn, n_pred, h_pred, f"Unseen: {title}")
    
    def plot_comparison(self, t_l, t_a, gt_n, gt_h, l_nn, a_nn, n_pred, h_pred, title):
        fig = plt.figure(figsize=(14, 6))
        
        # Panel 1: Physics
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(t_l.cpu().numpy().flatten(), t_a.cpu().numpy().flatten(), 'k-', lw=3, label="Target")
        ax1.plot(l_nn.cpu().numpy().flatten(), a_nn.cpu().numpy().flatten(), 'b--', lw=2, label="Prediction")
        ax1.set_title(f"Contact Law: {title}")
        ax1.set_xlabel("Load [N]")
        ax1.set_ylabel("Area [m²]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Parameters
        ax2 = plt.subplot(1, 2, 2)
        width = 0.35
        
        # Sort predictions
        sorted_idx = torch.argsort(h_pred[0])
        nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
        indices = np.arange(len(nn_h_sorted))
        
        if gt_h is not None:
             # If GT exists, align sorting
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

    def plot_triple_comparison(self, t_l, t_a, gt_n, gt_h, 
                             l_nn, a_nn, n_pred, h_pred,
                             l_ref, a_ref, n_ref, h_ref, title):
        """
        Plots Target vs NN vs Refined.
        """
        fig = plt.figure(figsize=(14, 6))
        
        # Panel 1: Physics Curves
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(t_l.cpu().numpy().flatten(), t_a.cpu().numpy().flatten(), 'k-', lw=3, label="Target (GT)")
        ax1.plot(l_nn.cpu().numpy().flatten(), a_nn.cpu().numpy().flatten(), 'b--', lw=2, label="Zero-Shot (NN)")
        ax1.plot(l_ref.cpu().numpy().flatten(), a_ref.cpu().numpy().flatten(), 'g:', lw=2, label="Refined (Opt)")
        
        ax1.set_title(f"Contact Law: {title}")
        ax1.set_xlabel("Load [N]")
        ax1.set_ylabel("Area [m²]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Parameters (Structure)
        ax2 = plt.subplot(1, 2, 2)
        width = 0.25
        
        # Sort by GT heights if available, else by prediction
        if gt_h is not None:
            sorted_idx = torch.argsort(gt_h[0])
            gt_h_sorted = gt_h[0][sorted_idx].cpu().numpy()
            nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
            ref_h_sorted = h_ref[0][sorted_idx].cpu().detach().numpy()
            indices = np.arange(len(gt_h_sorted))
            
            ax2.bar(indices - width, gt_h_sorted, width, label='Ground Truth', color='black', alpha=0.7)
            ax2.bar(indices, nn_h_sorted, width, label='NN Pred', color='blue', alpha=0.7)
            ax2.bar(indices + width, ref_h_sorted, width, label='Refined', color='green', alpha=0.7)
        else:
            # Fallback sort
            sorted_idx = torch.argsort(h_pred[0])
            nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
            ref_h_sorted = h_ref[0][sorted_idx].cpu().detach().numpy()
            indices = np.arange(len(nn_h_sorted))
            
            ax2.bar(indices - width/2, nn_h_sorted, width, label='NN Pred', color='blue', alpha=0.7)
            ax2.bar(indices + width/2, ref_h_sorted, width, label='Refined', color='green', alpha=0.7)

        ax2.set_title("Predicted Topography Structure")
        ax2.set_xlabel("Asperity Index")
        ax2.legend()
        
        plt.tight_layout()
        os.makedirs("plots_test", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else "sample"
        save_path = f"plots_test/val_test_{sname}.png"
        plt.savefig(save_path, dpi=150)
        print(f"[Validator] Saved plot to {save_path}")
        plt.close()

if __name__ == "__main__":
    val = UnifiedValidator("config.yaml")
    
    # Run the rigorous Test Set Validation
    # val.validate_on_test_set()
    # val.validate_designed(target_type="linear", refine=True)
    val.validate_designed(target_type="saturate", refine=True)
    # val.validate_designed(target_type="bilinear", refine=True)