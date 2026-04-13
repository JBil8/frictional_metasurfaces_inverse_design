import torch
import numpy as np
import sys
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer
from physics.tamas_solution import run_tamas_simulation
from utils.optimizer import refine_topology  # Ensure this utility is accessible

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'])
    X_all = data["x"]
    
    # Normalization constants (excluding -1.0 padding)
    GLOBAL_MAX_P = data["p_star_max"]
    valid_alpha = X_all[:, 1, :][X_all[:, 1, :] != -1.0]
    valid_stiff = X_all[:, 2, :][X_all[:, 2, :] != -1.0]
    GLOBAL_MAX_ALPHA = valid_alpha.max().item()
    GLOBAL_MAX_STIFF = valid_stiff.max().item()
    
    test_indices = [0, len(X_all) // 2, -1] 
    saved_results = {}

    model = SurfaceInverseModel(cfg).to(device)
    model.load_state_dict(torch.load(cfg['model']['name'], map_location=device))
    model.eval()

    phys_engine = AxisymmetricContactLayer(cfg=cfg).to(device)
    
    # Establish fixed grids for the optimizer
    max_indent = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
    n_steps = cfg['data']['n_steps']
    indents = torch.linspace(0, max_indent, n_steps).unsqueeze(0).to(device)
    # The grid the target was generated on
    p_star_grid = torch.linspace(0, GLOBAL_MAX_P, n_steps).to(device)

    for idx in test_indices:
        real_idx = idx if idx >= 0 else len(X_all) + idx
        print(f"\n=== Validating & Refining Sample Index {real_idx} ===")

        raw_x = X_all[idx].to(device)
        target_pressures_t = raw_x[0] # (Steps,)
        target_alphas_t = raw_x[1]    # (Steps,)

        # 1. Prepare NN Input & Zero-Shot Prediction
        nn_input = raw_x.clone().unsqueeze(0) 
        nn_input[:, 0, :] /= GLOBAL_MAX_P
        nn_input[:, 1, :] = torch.where(nn_input[:, 1, :] != -1.0, nn_input[:, 1, :] / GLOBAL_MAX_ALPHA, nn_input[:, 1, :])
        nn_input[:, 2, :] = torch.where(nn_input[:, 2, :] != -1.0, nn_input[:, 2, :] / GLOBAL_MAX_STIFF, nn_input[:, 2, :])
        
        with torch.no_grad():
            n_nn_t, h_nn_t = model(nn_input)
        
        # 2. RUN L-BFGS REFINEMENT (The "Polishing" Step)
        # We run 20 steps to snap the topography to the target curve
        w_t = torch.ones_like(n_nn_t) * (2.0 * cfg['physics']['radius'])
        
        n_opt_t, h_opt_t, p_opt_t, a_opt_t, _ = refine_topology(
            target_alpha=target_alphas_t,
            target_p=target_pressures_t,
            n_init=n_nn_t,
            h_init=h_nn_t,
            phys_engine=phys_engine,
            p_star_grid=p_star_grid,
            t_w=w_t,
            indentations=indents,
            k_steepness=1e5,
            steps=20 
        )

        # 3. Running Tamaas on the REFINED parameters
        # 3. Running Tamaas on the REFINED parameters
        pred_n = n_opt_t.detach().cpu().numpy()[0]
        pred_h = h_opt_t.detach().cpu().numpy()[0]
        
        valid_mask = (target_alphas_t != -1.0).cpu().numpy()
        valid_pressures = target_pressures_t.cpu().numpy()[valid_mask]

        # --- CRITICAL FIX: Interpolate Optimizer Output to P* Grid ---
        p_opt_np = p_opt_t.detach().cpu().numpy()[0]
        a_opt_np = a_opt_t.detach().cpu().numpy()[0]
        target_p_np = target_pressures_t.cpu().numpy()
        
        alpha_opt_aligned = np.interp(target_p_np, p_opt_np, a_opt_np, right=-1.0)
        # -------------------------------------------------------------

        print("  > Running Tamaas on Refined Surface...")
        bem_alphas = None
        bem_surface = None
        tamaas_pressure_steps = None

        try:
            # Solve for a subset of pressures to save time
            tamaas_pressure_steps = valid_pressures[::5] 
            
            bem_alphas, bem_surface, bem_L = run_tamas_simulation(
                heights=pred_h, 
                ns=pred_n, 
                widths=np.ones_like(pred_n) * (2.0 * cfg['physics']['radius']), 
                target_pressures=tamaas_pressure_steps,
                L=phys_engine.L, 
                E_star=cfg['physics']['E_star']
            )
            
            # Print quality metric
            min_pressure = tamaas_pressure_steps[1] 
            min_load = min_pressure * (phys_engine.L ** 2)
            est_min_radius = (0.75 * min_load * cfg['physics']['radius'] / cfg['physics']['E_star'])**(1/3)
            ppc = est_min_radius / (bem_L / bem_surface.shape[0])
            print(f"  > GRID QUALITY: {ppc:.1f} pixels per contact radius.")

        except Exception as e:
            print(f"  > Tamaas Failed: {e}")

        # 4. Save results with explicit naming for the optimized version
        saved_results[f"sample_{real_idx}"] = {
            "pressure_gt": target_p_np,     
            "alpha_gt": target_alphas_t.cpu().numpy(),           
            "alpha_nn_opt_analytical": alpha_opt_aligned, # <-- Correctly interpolated & renamed
            "pressure_bem": tamaas_pressure_steps, 
            "alpha_bem": bem_alphas,
            "params_pred_n": pred_n,
            "params_pred_h": pred_h,
            "surface_bem": bem_surface,
            "L_bem": phys_engine.L                                             
        }  

    os.makedirs("./data", exist_ok=True)
    np.savez("./data/paper_validation_data.npz", **saved_results)
    print("\nAll validation data saved to 'paper_validation_data.npz'")

if __name__ == "__main__":
    main()