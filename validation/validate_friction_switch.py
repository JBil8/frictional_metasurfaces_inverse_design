import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torch import nn

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.config import load_config
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer

def refine_prediction(self, target_stiff, n_init, h_init, indent_profile, steps=50):
    print(f"  > Refinement: Tandem Fine-Tuning (NN Init + L-BFGS)...")

    # 1. Setup trainable parameters from the CNN's exact output
    n_opt = n_init.clone().detach().requires_grad_(True)
    h_opt = h_init.clone().detach().requires_grad_(True)

    # L-BFGS is incredibly fast and accurate for local fine-tuning
    optimizer = optim.LBFGS([n_opt, h_opt], lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
    criterion = nn.MSELoss()

    # 2. Use an INTERMEDIATE steepness for gradients
    # Sharp enough to preserve the cliff location, smooth enough to provide a gradient
    k_fine_tune = 5e4 

    for i in range(steps):
        def closure():
            optimizer.zero_grad()
            
            # Enforce physical sorting during optimization
            h_sorted, _ = torch.sort(h_opt, dim=1)
            h_sorted = h_sorted - h_sorted[:, 0:1]

            # Run physics with the intermediate steepness
            _, _, s_pred = self.phys(
                h_sorted, 
                n_opt, 
                self.gen.t_w, 
                indent_profile, 
                k_steepness=k_fine_tune 
            )

            loss = criterion(s_pred / self.MAX_S, target_stiff / self.MAX_S)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"    [Warning] Optimization step failed: {e}")
            break

    # 3. Final Evaluation using TRUE, DISCONTINUOUS PHYSICS (k = 1e6)
    with torch.no_grad():
        h_final, _ = torch.sort(h_opt, dim=1)
        h_final = h_final - h_final[:, 0:1]
        n_final = n_opt
        
        p_final, alpha_final, s_final = self.phys(
            h_final, 
            n_final, 
            self.gen.t_w, 
            indent_profile, 
            k_steepness=1e6 # Hard physical check
        )

    return n_final, h_final, p_final, alpha_final, s_final

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