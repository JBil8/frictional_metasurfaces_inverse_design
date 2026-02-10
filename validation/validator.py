import torch
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

    def validate_on_test_set(self):
        """
        Plots one random sample from the Test Set for each category.
        Guaranteed to be data the network NEVER saw during training.
        """
        test_buckets = self.get_test_set_indices_by_category()
        
        for category, indices in test_buckets.items():
            if len(indices) == 0:
                print(f"Warning: No test samples found for {category}")
                continue
                
            # Pick one random index from the Test bucket
            random_test_idx = np.random.choice(indices)
            print(f"Validating {category.upper()} on Test Sample #{random_test_idx}...")
            
            # Fetch that specific sample
            t_l, t_a, gt_n, gt_h, title = self.gen.get_custom_sample(random_test_idx, category)
            
            # Predict
            prepend_val = torch.zeros(1, 1).to(self.device)
            raw_stiff = torch.diff(t_l, dim=1, prepend=prepend_val)
            
            nn_input = torch.cat([
                t_l / self.MAX_L, 
                t_a / self.MAX_A, 
                raw_stiff / self.MAX_S
            ], dim=0).unsqueeze(0)
            
            with torch.no_grad():
                n_pred, h_pred = self.model(nn_input)
                l_nn, a_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations)
            
            # Plot
            self.plot_comparison(t_l, t_a, gt_n, gt_h, l_nn, a_nn, n_pred, h_pred, f"Test Set: {category} (#{random_test_idx})")

    def validate_designed(self, target_type="linear"):
        """
        Validates on a purely synthetic target curve generated on the fly.
        Useful for testing "Unseen Physics" (Linear, Saturating, etc.).
        """
        print(f"[Validator] Generating fresh synthetic target: {target_type}...")
        
        # 1. Generate the Target Curve (Load vs Area)
        # Note: These targets do NOT have ground truth parameters (n, h)
        if target_type == "linear":
            t_l, t_a, title = self.gen.get_linear_coulomb()
        elif target_type == "saturate":
            t_l, t_a, title = self.gen.get_saturating_exponential()
        elif target_type == "bilinear":
            t_l, t_a, title = self.gen.get_bilinear_transition()
        elif target_type == "power":
            t_l, t_a, title = self.gen.get_power_law(exponent=1.5)
        elif target_type == "switch":
            t_l, t_a, title = self.gen.get_friction_switch()
        elif target_type == "step":
            t_l, t_a, title = self.gen.get_step_contact()
        else:
            raise ValueError(f"Unknown target type: {target_type}")

        # 2. Prepare Input for NN
        # Calculate stiffness
        prepend_val = torch.zeros(1, 1).to(self.device)
        raw_stiff = torch.diff(t_l, dim=1, prepend=prepend_val)
        
        # Normalize
        nn_input = torch.cat([
            t_l / self.MAX_L, 
            t_a / self.MAX_A, 
            raw_stiff / self.MAX_S
        ], dim=0).unsqueeze(0)
        
        # 3. NN Prediction
        with torch.no_grad():
            n_pred, h_pred = self.model(nn_input)
            
            # Reconstruct the curve from the predicted parameters
            l_nn, a_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations)
            
        # 4. Plot
        # We pass None for gt_n and gt_h because these synthetic curves 
        # don't have a "true" surface topography behind them.
        self.plot_comparison(t_l, t_a, None, None, l_nn, a_nn, n_pred, h_pred, f"Unseen: {title}")

    def plot_comparison(self, t_l, t_a, gt_n, gt_h, l_nn, a_nn, n_pred, h_pred, title):
        """
        Updated to handle cases where Ground Truth parameters (gt_n, gt_h) are missing.
        """
        fig = plt.figure(figsize=(14, 6))
        
        # --- Panel 1: Physics (Load vs Area) ---
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(t_l.cpu().numpy().flatten(), t_a.cpu().numpy().flatten(), 'k-', lw=3, label="Target (Synthetic)")
        ax1.plot(l_nn.cpu().numpy().flatten(), a_nn.cpu().numpy().flatten(), 'b--', lw=2, label="NN Prediction")
        
        ax1.set_title(f"Contact Law: {title}")
        ax1.set_xlabel("Load [N]")
        ax1.set_ylabel("Area [m²]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- Panel 2: Parameters (Bar Chart) ---
        ax2 = plt.subplot(1, 2, 2)
        width = 0.35
        
        # If we have Ground Truth (Real Data), plot comparison
        if gt_h is not None:
            sorted_idx = torch.argsort(gt_h[0])
            gt_h_sorted = gt_h[0][sorted_idx].cpu().numpy()
            nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
            indices = np.arange(len(gt_h_sorted))
            
            ax2.bar(indices - width/2, gt_h_sorted, width, label='Ground Truth', color='black', alpha=0.7)
            ax2.bar(indices + width/2, nn_h_sorted, width, label='NN Pred', color='blue', alpha=0.7)
        
        # If we DO NOT have Ground Truth (Synthetic Target), just plot the prediction
        else:
            # Sort by predicted height for readability
            sorted_idx = torch.argsort(h_pred[0])
            nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
            indices = np.arange(len(nn_h_sorted))
            
            ax2.bar(indices, nn_h_sorted, width, label='NN Pred (Inferred Structure)', color='blue', alpha=0.7)
            ax2.text(0.5, 0.9, "No GT: Synthetic Target", transform=ax2.transAxes, ha='center')

        ax2.set_title("Predicted Topography Structure")
        ax2.set_xlabel("Asperity Index (Sorted)")
        ax2.set_ylabel("Height Offset h [m]")
        ax2.legend()
        
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        sname = title.split(":")[1].strip().split(" ")[0].lower() if ":" in title else title.split(" ")[0].lower()
        save_path = f"plots/val_unseen_{sname}.png"
        plt.savefig(save_path, dpi=150)
        print(f"[Validator] Saved plot to {save_path}")
        plt.close()

if __name__ == "__main__":
    val = UnifiedValidator("config.yaml")
    
    # Run the rigorous Test Set Validation
    val.validate_on_test_set()
    val.validate_designed(target_type="linear")

    # Test if it can simulate "Bottoming out"
    val.validate_designed(target_type="saturate")

    # Test the transition
    val.validate_designed(target_type="bilinear")