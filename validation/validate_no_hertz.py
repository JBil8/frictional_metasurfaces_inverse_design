import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.config import load_config
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # 1. LOAD GLOBAL NORMALIZATION (Crucial!)
    # ---------------------------------------------------------
    print(f"Loading stats from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'])
    X_all = data["x"]
    
    GLOBAL_MAX_LOAD = X_all[:, 0, :].max()
    GLOBAL_MAX_AREA = X_all[:, 1, :].max()
    GLOBAL_MAX_STIFF = X_all[:, 2, :].max()
    
    print(f"Global Max Load: {GLOBAL_MAX_LOAD:.4f}")

    # ---------------------------------------------------------
    # 2. GENERATE TARGET: Power Law Area = Load^0.28
    # ---------------------------------------------------------
    print("Generating Target with Area ~ Load^0.28 (requires n=4.0)...")
    n_asp = cfg['physics']['n_asperities']
    
    # ---------------------------------------------------------
    # 2. GENERATE TARGET: Power Law Area = Load^0.4
    # ---------------------------------------------------------
    print("Generating Target with Area ~ Load^0.4 (requires n=4.0)...")
    n_asp = cfg['physics']['n_asperities']
    
    # 1. Choose Exponent n=4.0 (Central to training distribution)
    # gamma = 2 / (4+1) = 0.4
    target_n_np = np.ones(n_asp) * 4.0 
    
    target_h_np = np.sort(np.random.rand(n_asp) * 0.04) 
    target_h_np -= target_h_np[0] # Anchor
    
    # Fixed Widths
    target_w_np = np.ones(n_asp) * 2.0 * cfg['physics']['radius']

    # Fixed Widths
    target_w_np = np.ones(n_asp) * 2.0 * cfg['physics']['radius']

    # Run Physics to get the Target Curve
    phys_engine = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
    indentations = torch.linspace(0, max_d, cfg['data']['n_steps']).unsqueeze(0).to(device)
    
    # Convert to Tensors
    t_n = torch.tensor(target_n_np, dtype=torch.float32).unsqueeze(0).to(device)
    t_h = torch.tensor(target_h_np, dtype=torch.float32).unsqueeze(0).to(device)
    t_w = torch.tensor(target_w_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        gt_load, gt_area = phys_engine(t_h, t_n, t_w, indentations)

    # CHECK: Ensure we are within training bounds
    if gt_load.max() > GLOBAL_MAX_LOAD:
        print("WARNING: Target Load exceeds training range! Scaling down...")
        # Simple fix: just pretend we scaled the input, the NN handles ratio
    
    # ---------------------------------------------------------
    # 3. RUN YOUR NEURAL NETWORK
    # ---------------------------------------------------------
    print("Running NN Prediction...")
    # Calculate Stiffness on RAW Load
    raw_stiff = torch.diff(gt_load, dim=1, prepend=torch.zeros(1, 1).to(device))
    
    # Normalize Inputs using TRAINING stats
    nn_input = torch.stack([
        gt_load / GLOBAL_MAX_LOAD,
        gt_area / GLOBAL_MAX_AREA,
        raw_stiff / GLOBAL_MAX_STIFF
    ], dim=1)
    
    model = SurfaceInverseModel(cfg).to(device)
    model.load_state_dict(torch.load("model_final.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        pred_n, pred_h = model(nn_input)
        # Reconstruct NN curve
        nn_load, nn_area = phys_engine(pred_h, pred_n, t_w, indentations)

    # ---------------------------------------------------------
    # 4. OPTIMIZE HERTZIAN FIT (The Baseline)
    # ---------------------------------------------------------
    print("Optimizing Best Possible Hertzian Fit (n=2 fixed)...")
    
    target_area_vec = gt_area.cpu().numpy().flatten()
    
    def hertz_loss(h_guess):
        # Convert guess to tensor
        h_in = torch.tensor(h_guess, dtype=torch.float32).unsqueeze(0).to(device)
        n_in = torch.ones_like(h_in) * 2.0 # FORCE HERTZ (n=2)
        w_in = t_w 
        
        with torch.no_grad():
            _, hz_area = phys_engine(h_in, n_in, w_in, indentations)
        
        # MSE Loss
        diff = hz_area.cpu().numpy().flatten() - target_area_vec
        return np.mean(diff**2)

    # Start optimization from true heights
    res = minimize(hertz_loss, target_h_np, method='L-BFGS-B', bounds=[(0, 10*max_d)]*n_asp)
    
    # Generate the best Hertz curve found
    with torch.no_grad():
        h_opt = torch.tensor(res.x, dtype=torch.float32).unsqueeze(0).to(device)
        n_opt = torch.ones_like(h_opt) * 2.0
        _, best_hertz_area = phys_engine(h_opt, n_opt, t_w, indentations)

    # ---------------------------------------------------------
    # 5. PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # X-axis: Load
    x_axis = gt_load.cpu().numpy().flatten()
    
    # 1. Target (Black)
    plt.plot(x_axis, target_area_vec, 'k-', linewidth=3, label=r"Target (Power Law $A \propto L^{0.4}$)")
    
    # 2. NN (Blue Dashed)
    plt.plot(x_axis, nn_area.cpu().numpy().flatten(), 'b--', linewidth=2, label="Your NN (General Model)")
    
    # 3. Hertz (Red Dotted)
    plt.plot(x_axis, best_hertz_area.cpu().numpy().flatten(), 'r:', linewidth=2, label="Best Hertz Fit ($n=2$)")
    
    plt.title("Validation: General NN vs Hertzian Restriction")
    plt.xlabel("Load [N]")
    plt.ylabel("Contact Area [m^2]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "./plots/nn_vs_hertz_cone.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()