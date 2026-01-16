import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.config import load_config
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer
from physics.tamas_solution import run_tamas_simulation

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the SAME dataset used for training
    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'])
    X_all = data["x"]  # Raw Data (N, 3, Steps)
    Y_all = data["y"]  # Raw Params (N, Params)

    # 2. Calculate GLOBAL Normalization Constants (Exactly like Main)
    # This ensures 100% consistency with how the model learned.
    GLOBAL_MAX_LOAD = X_all[:, 0, :].max()
    GLOBAL_MAX_AREA = X_all[:, 1, :].max()
    GLOBAL_MAX_STIFF = X_all[:, 2, :].max()
    
    print(f"Global Max Load: {GLOBAL_MAX_LOAD:.4f}")
    print(f"Global Max Stiff: {GLOBAL_MAX_STIFF:.4f}")

    # 3. Pick a Test Sample
    # We pick the last sample (-1) to simulate "unseen" data
    sample_idx = -1 
    
    # Get Raw Curves (for plotting/physics) and Raw Params
    raw_x = X_all[sample_idx].to(device) # (3, Steps)
    gt_params = Y_all[sample_idx].to(device) # (32,)

    # Extract GT parameters
    n_asp = cfg['physics']['n_asperities']
    gt_n = gt_params[:n_asp].cpu().numpy()
    gt_h = gt_params[n_asp:].cpu().numpy()
    
    # We need the load curve for Tamaas input (Steps)
    # Channel 0 is Load
    target_loads = raw_x[0].cpu().numpy()

    # 4. Prepare Input for NN (Normalize using GLOBAL constants)
    # We must reshape to (1, 3, Steps) for the model
    nn_input = raw_x.clone().unsqueeze(0) 
    nn_input[:, 0, :] /= GLOBAL_MAX_LOAD
    nn_input[:, 1, :] /= GLOBAL_MAX_AREA
    nn_input[:, 2, :] /= GLOBAL_MAX_STIFF

    # 5. Run Prediction
    print("Running Neural Network Prediction...")
    model = SurfaceInverseModel(cfg).to(device)
    model.load_state_dict(torch.load("model_final.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        pred_n_t, pred_h_t = model(nn_input)
    
    pred_n = pred_n_t.cpu().numpy()[0]
    pred_h = pred_h_t.cpu().numpy()[0]

    # 6. Run Tamaas BEM Validation
    print("\nRunning TAMAAS BEM Simulation...")
    try:
        max_indent_phys = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        
        # We assume Widths are fixed/known (from config logic)
        R = cfg['physics']['radius']
        gt_w = np.ones_like(gt_n) * (2.0 * R)

        bem_areas, bem_surface, bem_L = run_tamas_simulation(
            heights=pred_h, 
            ns=pred_n, 
            widths=gt_w, 
            target_loads=target_loads[::10],
            max_indentation=max_indent_phys,
            E_star=cfg['physics']['E_star']
        )
        bem_success = True
    except Exception as e:
        print(f"Tamaas Error: {e}")
        bem_success = False
        bem_areas = None

    # 7. Plotting
    print("Plotting results...")
    fig = plt.figure(figsize=(16, 7))
    
    # Subplot 1: Curves
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Plot Raw GT Area (Channel 1)
    gt_area_curve = raw_x[1].cpu().numpy()
    ax1.plot(target_loads, gt_area_curve, 'k-', linewidth=2.5, label='Target (Data GT)')
    
    # Plot NN Prediction (Analytical check)
    phys_engine = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    indentations = torch.linspace(0, max_indent_phys, cfg['data']['n_steps']).unsqueeze(0).to(device)
    with torch.no_grad():
        # Reconstruct using predicted parameters
        _, pred_area_analytical = phys_engine(pred_h_t, pred_n_t, torch.tensor(gt_w).unsqueeze(0).to(device), indentations)
    ax1.plot(target_loads, pred_area_analytical.cpu().numpy()[0], 'b--', linewidth=2, label='NN Prediction (Analytical)')
    
    if bem_success:
        ax1.plot(target_loads[::10], bem_areas, 'r.', markersize=8, label='BEM Verification')

    ax1.set_xlabel("Load")
    ax1.set_ylabel("Area")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: 3D Surface
    if bem_success and bem_surface is not None:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        N = bem_surface.shape[0]
        x = np.linspace(0, bem_L, N)
        y = np.linspace(0, bem_L, N)
        X, Y = np.meshgrid(x, y)
        # clip bem_surface for better visualization
        bem_surface = np.clip(bem_surface, -1.0, 0.0)
        surf = ax2.plot_surface(X, Y, bem_surface, cmap='coolwarm', linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax2, shrink=0.5)
        ax2.set_title(f"Predicted Surface (L={bem_L:.2f})")

    plt.tight_layout()
    plt.savefig("validation_bem_fixed.png")
    plt.show()

if __name__ == "__main__":
    main()