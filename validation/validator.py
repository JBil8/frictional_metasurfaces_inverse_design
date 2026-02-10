import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    def validate_sample(self, category="switch", offset=5):
        """
        Validates on a real sample using EXACT INDICES.
        Requires dataset_16_asp_mixed.pt to be generated correctly.
        """
        # DIRECT FETCH: No scanning. We trust the generator.
        t_l, t_a, gt_n, gt_h, title = self.gen.get_dataset_sample(category, offset)
        
        # NN Prediction
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
            
        self.plot_comparison(t_l, t_a, gt_n, gt_h, l_nn, a_nn, n_pred, h_pred, title)

    def plot_comparison(self, t_l, t_a, gt_n, gt_h, l_nn, a_nn, n_pred, h_pred, title):
        fig = plt.figure(figsize=(14, 6))
        
        # Panel 1: Physics
        ax1 = plt.subplot(1, 2, 1)
        # Plot Curves
        ax1.plot(t_l.cpu().numpy().flatten(), t_a.cpu().numpy().flatten(), 'k-', lw=3, label="Ground Truth")
        ax1.plot(l_nn.cpu().numpy().flatten(), a_nn.cpu().numpy().flatten(), 'b--', lw=2, label="NN Prediction")
        
        ax1.set_title(f"Contact Law: {title}")
        ax1.set_xlabel("Load [N]")
        ax1.set_ylabel("Area [m²]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Parameters (The Truth Test)
        ax2 = plt.subplot(1, 2, 2)
        width = 0.35
        
        # Sort by GT height to reveal structure
        # If 'Switch', we should see two groups of bars
        # If 'Sparse', we should see a gap on the left
        sorted_idx = torch.argsort(gt_h[0])
        gt_h_sorted = gt_h[0][sorted_idx].cpu().numpy()
        nn_h_sorted = h_pred[0][sorted_idx].cpu().detach().numpy()
        
        indices = np.arange(len(gt_h_sorted))
        
        ax2.bar(indices - width/2, gt_h_sorted, width, label='Ground Truth', color='black', alpha=0.7)
        ax2.bar(indices + width/2, nn_h_sorted, width, label='NN Pred', color='blue', alpha=0.7)
        
        ax2.set_title("Topography Structure (Sorted by Height)")
        ax2.set_xlabel("Asperity Index")
        ax2.set_ylabel("Height Offset h [m]")
        ax2.legend()
        
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        # Clean filename
        sname = title.split(":")[1].strip().split(" ")[0].lower() + f"_{title.split('#')[1][:-1]}"
        save_path = f"plots/val_fixed_{sname}.png"
        plt.savefig(save_path, dpi=150)
        print(f"[Validator] Saved plot to {save_path}")
        plt.close()

if __name__ == "__main__":
    val = UnifiedValidator("config.yaml")
    
    print("--- Running Validation on Mixed Dataset ---")
    val.validate_sample(category="switch", offset=10)
    val.validate_sample(category="sparse", offset=10)
    val.validate_sample(category="wall", offset=5)
    val.validate_sample(category="lhs", offset=5)