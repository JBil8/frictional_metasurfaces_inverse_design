import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.config import load_config
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer

def refine_prediction(phys_engine, target_load, target_area, init_n, init_h, 
                      t_w, indentations, steps=500, lr=0.002): # More steps, lower LR
    """
    Refines prediction using Projected Gradient Descent and L1 Loss.
    """
    print(f"  > Starting Refinement ({steps} steps)...")
    
    # 1. Clone and Detach
    opt_n = init_n.clone().detach().requires_grad_(True)
    opt_h = init_h.clone().detach().requires_grad_(True)
    
    # 2. Optimizer
    optimizer = optim.Adam([opt_n, opt_h], lr=lr)
    
    # scheduler to slow down as we get closer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    
    loss_history = []
    
    for i in range(steps):
        optimizer.zero_grad()
        
        # 3. Forward (Use the variables directly)
        rec_l, rec_a = phys_engine(opt_h, opt_n, t_w, indentations)
        
        # 4. Robust Loss (L1 Loss is better for sharp kinks)
        # We focus purely on the Area-Load relationship
        loss_area = torch.nn.functional.l1_loss(rec_a, target_area)
        loss_load = torch.nn.functional.l1_loss(rec_l, target_load)
        
        total_loss = loss_area * 10 + loss_load * 100 # Stronger weight on Area shape
        
        total_loss.backward()
        optimizer.step()
        
        # 5. PROJECTED GRADIENT STEP (Crucial Fix!)
        # Clamp the actual variables in-place so they never go unphysical
        with torch.no_grad():
            opt_n.data.clamp_(1.0, 3.0)
            opt_h.data.clamp_(0.0, indentations.max().item())
            
            # Optional: Enforce sorting to help the optimizer? 
            # No, let it find its own way, but sorting helps visualization later.
        
        scheduler.step(total_loss)
        loss_history.append(total_loss.item())
        
        if i % 100 == 0:
            print(f"    Step {i}: Loss = {total_loss.item():.6f}")

    return opt_n.detach(), opt_h.detach(), loss_history
def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ... [Load Data & Normalization as before] ...
    print(f"Loading stats from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'])
    X_all = data["x"]
    
    GLOBAL_MAX_LOAD = X_all[:, 0, :].max().item()
    GLOBAL_MAX_AREA = X_all[:, 1, :].max().item()
    GLOBAL_MAX_STIFF = X_all[:, 2, :].max().item()

    # ... [Generate Friction Switch Target as before] ...
    print("Generating Synthetic 'Friction Switch' Target...")
    n_steps = cfg['data']['n_steps']
    steps_norm = torch.linspace(0, 1, n_steps).to(device)
    target_load = 0.25 * GLOBAL_MAX_LOAD * (steps_norm ** 1.5)
    
    P_crit = 0.1 * GLOBAL_MAX_LOAD  
    slope_slip = 0.55 * (GLOBAL_MAX_AREA / GLOBAL_MAX_LOAD) 
    slope_lock = 1.50 * (GLOBAL_MAX_AREA / GLOBAL_MAX_LOAD)
    
    target_area = torch.zeros_like(target_load)
    mask_slip = target_load < P_crit
    mask_lock = ~mask_slip
    target_area[mask_slip] = target_load[mask_slip] * slope_slip
    A_crit = P_crit * slope_slip
    target_area[mask_lock] = A_crit + (target_load[mask_lock] - P_crit) * slope_lock

    # ---------------------------------------------------------
    # 3. INITIAL NN PREDICTION
    # ---------------------------------------------------------
    print("Running NN Prediction (Initial Guess)...")
    raw_stiff = torch.diff(target_load, prepend=torch.tensor([0.0]).to(device))
    nn_input = torch.stack([
        target_load / GLOBAL_MAX_LOAD,
        target_area / GLOBAL_MAX_AREA,
        raw_stiff / GLOBAL_MAX_STIFF
    ], dim=0).unsqueeze(0)
    
    model = SurfaceInverseModel(cfg).to(device)
    model.load_state_dict(torch.load("model_final.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        pred_n_init, pred_h_init = model(nn_input)
    
    # Physics Setup
    phys_engine = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    n_asp = cfg['physics']['n_asperities']
    t_w = torch.ones(1, n_asp).to(device) * 2.0 * cfg['physics']['radius']
    max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
    indentations = torch.linspace(0, max_d, n_steps).unsqueeze(0).to(device)

    # ---------------------------------------------------------
    # 4. TEST-TIME REFINEMENT (The "Active Learning" Step)
    # ---------------------------------------------------------
    print("Running Physics-Guided Refinement...")
    
    # We pass the NN output as the starting point
    pred_n_opt, pred_h_opt, history = refine_prediction(
        phys_engine, target_load, target_area, 
        pred_n_init, pred_h_init, 
        t_w, indentations, 
        steps=200, lr=0.005  # 200 steps is usually enough
    )

    # ---------------------------------------------------------
    # 5. FINAL RECONSTRUCTION & PLOTTING
    # ---------------------------------------------------------
    # Get curves for Initial Guess (NN)
    with torch.no_grad():
        l_init, a_init = phys_engine(pred_h_init, pred_n_init, t_w, indentations)
        
    # Get curves for Optimized Result (Refined)
    with torch.no_grad():
        l_opt, a_opt = phys_engine(pred_h_opt, pred_n_opt, t_w, indentations)

    plt.figure(figsize=(12, 6))
    
    # Plot 1: The Curves
    plt.subplot(1, 2, 1)
    plt.plot(target_load.cpu().numpy(), target_area.cpu().numpy(), 'k-', lw=3, label="Target (Switch)")
    plt.plot(l_init.cpu().numpy().flatten(), a_init.cpu().numpy().flatten(), 'b--', lw=2, label="NN Prediction (Zero-Shot)")
    plt.plot(l_opt.cpu().numpy().flatten(), a_opt.cpu().numpy().flatten(), 'g-', lw=2.5, label="Refined (Test-Time Opt)")
    
    plt.title("Performance of Physics-Guided Refinement")
    plt.xlabel("Normal Load [N]")
    plt.ylabel("Contact Area [m²]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: The Parameters (Showing the Change)
    plt.subplot(1, 2, 2)
    x = np.arange(n_asp)
    
    # Sort for visualization
    h_init_sorted, _ = torch.sort(pred_h_init.flatten())
    h_opt_sorted, _ = torch.sort(pred_h_opt.flatten())
    
    plt.bar(x - 0.2, h_init_sorted.cpu().numpy(), width=0.4, label="NN Initial", color='blue', alpha=0.6)
    plt.bar(x + 0.2, h_opt_sorted.cpu().numpy(), width=0.4, label="Refined", color='green', alpha=0.8)
    
    plt.title("Evolution of Height Distribution")
    plt.xlabel("Asperity Index")
    plt.ylabel("Height Offset h [m]")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("refined_friction_switch.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()