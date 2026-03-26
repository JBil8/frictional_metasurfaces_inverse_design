import torch
import numpy as np
import sys
import os

# Set threads BEFORE imports
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

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'])
    X_all = data["x"]
    Y_all = data["y"]

    # Global Normalization
    GLOBAL_MAX_STIFF = X_all[:, 2, :].max()
    
    test_indices = [0, len(X_all) // 2, -1] 
    saved_results = {}

    model = SurfaceInverseModel(cfg).to(device)
    model.load_state_dict(torch.load(cfg['model']['name'], map_location=device))
    model.eval()

    # CRITICAL FIX 1: Pass the full config so the engine can calculate the global L
    phys_engine = AxisymmetricContactLayer(cfg=cfg).to(device)

    for idx in test_indices:
        real_idx = idx if idx >= 0 else len(X_all) + idx
        print(f"\n=== Validating Sample Index {real_idx} ===")

        raw_x = X_all[idx].to(device)
        
        # CRITICAL FIX 2: These are now intensive properties
        target_pressures = raw_x[0].cpu().numpy()
        target_alphas = raw_x[1].cpu().numpy() 

        nn_input = raw_x[2:3].clone().unsqueeze(0) 
        nn_input /= GLOBAL_MAX_STIFF
        
        with torch.no_grad():
            pred_n_t, pred_h_t = model(nn_input)
            
            max_indent = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
            indents = torch.linspace(0, max_indent, cfg['data']['n_steps']).unsqueeze(0).to(device)
            w_t = torch.ones_like(pred_n_t) * (2.0 * cfg['physics']['radius'])
            
            # CRITICAL FIX 3: Unpack Pressure and Alpha
            pred_p_analytical, pred_alpha_analytical, _ = phys_engine(pred_h_t, pred_n_t, w_t, indents)
        
        pred_n = pred_n_t.cpu().numpy()[0]
        pred_h = pred_h_t.cpu().numpy()[0]
        
        print("  > Running Tamaas...")
        try:
            tamaas_pressure_steps = target_pressures[::20] 
            
            # CRITICAL FIX 4: Call Tamaas with Intensive Pressures and Global L
            bem_alphas, bem_surface, bem_L = run_tamas_simulation(
                heights=pred_h, 
                ns=pred_n, 
                widths=np.ones_like(pred_n) * (2.0 * cfg['physics']['radius']), 
                target_pressures=tamaas_pressure_steps,
                L=phys_engine.L, # Pass the global L calculated by the PyTorch engine
                E_star=cfg['physics']['E_star']
            )
            
            N_pixels = bem_surface.shape[0]
            pixel_size = bem_L / N_pixels
            
            # Convert pressure back to load purely for the heuristic safety check
            min_pressure = tamaas_pressure_steps[1] 
            min_load = min_pressure * (phys_engine.L ** 2)
            est_min_radius = (0.75 * min_load * cfg['physics']['radius'] / cfg['physics']['E_star'])**(1/3)
            
            ppc = est_min_radius / pixel_size
            print(f"  > GRID QUALITY: {ppc:.1f} pixels per contact radius (at min load).")
            if ppc < 5.0:
                print("    [WARNING] Resolution might be too low for initial contact!")
            else:
                print("    [OK] Resolution is sufficient.")

        except Exception as e:
            print(f"  > Tamaas Failed: {e}")
            bem_alphas = None
            tamaas_pressure_steps = None

        # CRITICAL FIX 5: Save with Intensive Dictionary Keys
        saved_results[f"sample_{real_idx}"] = {
            "pressure_gt": target_pressures,
            "alpha_gt": target_alphas,
            "alpha_nn_analytical": pred_alpha_analytical.cpu().numpy()[0],
            "pressure_bem": tamaas_pressure_steps,
            "alpha_bem": bem_alphas,
            "params_pred_n": pred_n,
            "params_pred_h": pred_h
        }

    np.savez("./data/paper_validation_data.npz", **saved_results)
    print("\nAll validation data saved to 'paper_validation_data.npz'")

if __name__ == "__main__":
    main()