import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# Adjust path to import from parent directories
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
        # Load Config
        # We assume cfg_path is relative to the script running this, or absolute.
        # If running from 'validation/', config is '../config.yaml'
        if not os.path.exists(cfg_path):
             # Try parent dir if not found (common issue when running script from subdir)
             cfg_path = os.path.join("..", cfg_path)
             
        self.cfg = load_config(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[UnifiedValidator] Running on {self.device}")
        
        # 1. Load Data Statistics for Normalization
        print(f"[UnifiedValidator] Loading normalization stats...")
        data_path = self.cfg['data']['path']
        if not os.path.exists(data_path):
             data_path = os.path.join("..", data_path)
             
        data = torch.load(data_path, map_location=self.device)
        X_all = data["x"]
        limits = get_theoretical_limits(self.cfg, self.device)
        self.MAX_L = limits['max_load']
        self.MAX_A = limits['max_area']
        self.MAX_S = limits['max_stiff']
        
        # 2. Load Physics Engine
        self.phys = AxisymmetricContactLayer(E_star=self.cfg['physics']['E_star']).to(self.device)
        
        # 3. Load Neural Network
        self.model = SurfaceInverseModel(self.cfg).to(self.device)
        
        # Try finding the model file
        model_path = "model_final.pth"
        if not os.path.exists(model_path): model_path = "../model_final.pth"
        
        print(f"[UnifiedValidator] Loading model from {model_path}...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 4. Initialize Target Generator
        self.gen = TargetGenerator(self.phys, self.cfg, self.device)

    def run_refinement(self, target_l, target_a, init_n, init_h, steps=50):
        print(f"  [Refinement] Running L-BFGS Optimization...")
        
        opt_n = init_n.clone().detach().requires_grad_(True)
        opt_h = init_h.clone().detach().requires_grad_(True)
        
        # L-BFGS is powerful but requires a closure function
        optimizer = optim.LBFGS([opt_n, opt_h], lr=0.5, max_iter=20, history_size=20)
        
        t_w = self.gen.t_w
        ind = self.gen.indentations
        
        def closure():
            optimizer.zero_grad()
            
            # Clamp variables directly inside closure to keep them physical during line search
            with torch.no_grad():
                opt_n.data.clamp_(1.0, 8.0)
                opt_h.data.clamp_(0.0, ind.max().item())
            
            rec_l, rec_a = self.phys(opt_h, opt_n, t_w, ind)
            
            loss_area = torch.nn.functional.l1_loss(rec_a, target_a)
            loss_load = torch.nn.functional.l1_loss(rec_l, target_l)
            
            # Stronger weight on Area matching
            total_loss = loss_area * 50.0 + loss_load
            
            total_loss.backward()
            return total_loss

        # Run optimization steps
        for i in range(steps):
            loss = optimizer.step(closure)
            if i % 10 == 0:
                print(f"    L-BFGS Step {i}: Loss {loss.item():.6f}")
                
        return opt_n.detach(), opt_h.detach(), []

    def validate(self, target_type="switch", active_learning=True, save_plot=True):
        print(f"\n--- Starting Validation: {target_type} (Active Learning: {active_learning}) ---")
        
        # 1. Generate Target
        if target_type == "switch":
            t_l, t_a, title = self.gen.get_friction_switch()
        elif target_type == "power":
            t_l, t_a, title = self.gen.get_power_law(exponent=1.5)
        elif target_type == "step":
            t_l, t_a, title = self.gen.get_step_contact()
        else:
            raise ValueError(f"Unknown target type: {target_type}")
            
        # 2. NN Prediction (Zero-Shot)
        # Ensure shapes match for concatenation
        # t_l is [1, 500]. We want to stack them into [1, 3, 500]
        
        # Calculate stiffness (prepend to keep shape [1, 500])
        prepend_val = torch.zeros(1, 1).to(self.device)
        raw_stiff = torch.diff(t_l, dim=1, prepend=prepend_val)
        
        # Normalize
        norm_l = t_l / self.MAX_L
        norm_a = t_a / self.MAX_A
        norm_s = raw_stiff / self.MAX_S
        
        # Use torch.cat(dim=0) to merge [1, 500], [1, 500], [1, 500] -> [3, 500]
        # Then unsqueeze to get [1, 3, 500]
        nn_input = torch.cat([norm_l, norm_a, norm_s], dim=0).unsqueeze(0)
        
        with torch.no_grad():
            n_pred, h_pred = self.model(nn_input)
            # Reconstruct what the NN predicted using physics
            l_nn, a_nn = self.phys(h_pred, n_pred, self.gen.t_w, self.gen.indentations)
            
        # 3. Active Learning (Optional)
        l_opt, a_opt = None, None
        h_opt, n_opt = None, None
        
        if active_learning:
            n_opt, h_opt, _ = self.run_refinement(t_l, t_a, n_pred, h_pred)
            with torch.no_grad():
                l_opt, a_opt = self.phys(h_opt, n_opt, self.gen.t_w, self.gen.indentations)
            
        # 4. Plotting
        if save_plot:
            self.plot_results(t_l, t_a, l_nn, a_nn, l_opt, a_opt, title, active_learning)
            
        return {
            "target": (t_l, t_a),
            "nn": (l_nn, a_nn, n_pred, h_pred),
            "opt": (l_opt, a_opt, n_opt, h_opt) if active_learning else None
        }

    def plot_results(self, t_l, t_a, l_nn, a_nn, l_opt, a_opt, title, show_opt):
        plt.figure(figsize=(12, 5))
        
        # Panel 1: Physics Curves
        plt.subplot(1, 2, 1)
        t_l_np, t_a_np = t_l.cpu().numpy().flatten(), t_a.cpu().numpy().flatten()
        nn_l_np, nn_a_np = l_nn.cpu().numpy().flatten(), a_nn.cpu().numpy().flatten()
        
        plt.plot(t_l_np, t_a_np, 'k-', lw=3, label="Target")
        plt.plot(nn_l_np, nn_a_np, 'b--', lw=2, label="NN Prediction")
        
        if show_opt and l_opt is not None:
            opt_l_np, opt_a_np = l_opt.cpu().numpy().flatten(), a_opt.cpu().numpy().flatten()
            plt.plot(opt_l_np, opt_a_np, 'g-', lw=2.5, label="Refined")
            
        plt.title(f"{title}")
        plt.xlabel("Normal Load [N]")
        plt.ylabel("Real Contact Area [m²]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Panel 2: Physical Envelope (Debug)
        # Helps see if target was reasonable
        plt.subplot(1, 2, 2)
        env_l_min = self.gen.l_min.cpu().numpy().flatten()
        env_a_min = self.gen.a_min.cpu().numpy().flatten()
        env_l_max = self.gen.l_max.cpu().numpy().flatten()
        env_a_max = self.gen.a_max.cpu().numpy().flatten()
        
        plt.fill_between(env_l_max, env_a_min, env_a_max, color='gray', alpha=0.1, label="Feasible Envelope")
        plt.plot(t_l_np, t_a_np, 'r-', label="Target Path")
        plt.title("Feasibility Check")
        plt.xlabel("Load")
        plt.yticks([])
        plt.legend()

        plt.tight_layout()
        
        # Save to plots directory
        os.makedirs("/plots", exist_ok=True)
        sname = title.split(" ")[0].lower()
        save_path = f"/plots/val_{sname}.png"
        plt.savefig(save_path, dpi=150)
        print(f"[UnifiedValidator] Plot saved to {save_path}")
        plt.close()

# Example usage if run directly
if __name__ == "__main__":
    val = UnifiedValidator("config.yaml")
    val.validate(target_type="step", active_learning=True)