import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Set threads BEFORE imports
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.config import load_config
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer
from physics.tamas_solution import run_tamas_simulation

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'])
    X_all = data["x"]
    Y_all = data["y"]

    # Global Normalization
    GLOBAL_MAX_LOAD = X_all[:, 0, :].max()
    GLOBAL_MAX_AREA = X_all[:, 1, :].max()
    GLOBAL_MAX_STIFF = X_all[:, 2, :].max()
    
    # --- 1. Define Samples to Validate ---
    test_indices = [0, len(X_all) // 2, -1] # First, Middle, Last
    
    # Dictionary to store all results for saving
    saved_results = {}

    model = SurfaceInverseModel(cfg).to(device)
    model_name = cfg['model']['name']
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    # Create analytical engine for comparison
    phys_engine = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)

    for idx in test_indices:
        real_idx = idx if idx >= 0 else len(X_all) + idx
        print(f"\n=== Validating Sample Index {real_idx} ===")

        # A. Prepare Data
        raw_x = X_all[idx].to(device)
        gt_params = Y_all[idx].to(device)
        
        target_loads = raw_x[0].cpu().numpy()
        target_areas = raw_x[1].cpu().numpy() # For saving

        # B. NN Prediction
        nn_input = raw_x.clone().unsqueeze(0)
        nn_input[:, 0, :] /= GLOBAL_MAX_LOAD
        nn_input[:, 1, :] /= GLOBAL_MAX_AREA
        nn_input[:, 2, :] /= GLOBAL_MAX_STIFF
        
        with torch.no_grad():
            pred_n_t, pred_h_t = model(nn_input)
            
            # Analytical NN Check
            max_indent = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
            indents = torch.linspace(0, max_indent, cfg['data']['n_steps']).unsqueeze(0).to(device)
            # Assume constant widths for now (as per your training logic)
            w_t = torch.ones_like(pred_n_t) * (2.0 * cfg['physics']['radius'])
            _, pred_area_analytical = phys_engine(pred_h_t, pred_n_t, w_t, indents)
        
        pred_n = pred_n_t.cpu().numpy()[0]
        pred_h = pred_h_t.cpu().numpy()[0]
        
        # C. Tamaas Simulation
        print("  > Running Tamaas...")
        try:
            # Downsample loads for Tamaas (speedup)
            tamaas_load_steps = target_loads[::20] 
            
            bem_areas, bem_surface, bem_L = run_tamas_simulation(
                heights=pred_h, 
                ns=pred_n, 
                widths=np.ones_like(pred_n) * (2.0 * cfg['physics']['radius']), 
                target_loads=tamaas_load_steps,
                max_indentation=max_indent,
                E_star=cfg['physics']['E_star']
            )
            
            # Estimate pixel size
            N_pixels = bem_surface.shape[0]
            pixel_size = bem_L / N_pixels
            
            # Estimate Hertzian radius at lowest non-zero load for the sharpest asperity (n approx 2)
            min_load = tamaas_load_steps[1] # Skip 0
            # a approx (3FL/4E)^(1/3) for sphere. 
            # This is a rough heuristic to check safety.
            est_min_radius = (0.75 * min_load * cfg['physics']['radius'] / cfg['physics']['E_star'])**(1/3)
            
            ppc = est_min_radius / pixel_size
            print(f"  > GRID QUALITY: {ppc:.1f} pixels per contact radius (at min load).")
            if ppc < 5.0:
                print("    [WARNING] Resolution might be too low for initial contact!")
            else:
                print("    [OK] Resolution is sufficient.")

        except Exception as e:
            print(f"  > Tamaas Failed: {e}")
            bem_areas = None
            tamaas_load_steps = None

        # D. Store Data for Plotting Later
        saved_results[f"sample_{real_idx}"] = {
            "load_gt": target_loads,
            "area_gt": target_areas,
            "area_nn_analytical": pred_area_analytical.cpu().numpy()[0],
            "load_bem": tamaas_load_steps,
            "area_bem": bem_areas,
            "params_pred_n": pred_n,
            "params_pred_h": pred_h
        }

    # --- 4. Save to Disk ---
    # This creates one file you can load on your laptop to make perfect plots
    np.savez("./data/paper_validation_data.npz", **saved_results)
    print("\nAll validation data saved to 'paper_validation_data.npz'")

if __name__ == "__main__":
    main()