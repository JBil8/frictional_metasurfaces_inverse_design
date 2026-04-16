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
from utils.optimizer import refine_topology
from utils.interpolation import batched_interp1d

try:
    from validation.targets import TargetGenerator
except ImportError:
    from targets import TargetGenerator

def prepare_nn_input(t_p, t_a, t_s, p_grid, max_p, max_a, max_s):
    """Aligns any arbitrary target curve to the rigid CNN input grid."""
    p_2d, a_2d, s_2d = t_p.view(1, -1), t_a.view(1, -1), t_s.view(1, -1)
    
    aligned_a = batched_interp1d(p_grid, p_2d, a_2d, pad_value=-1.0)
    aligned_s = batched_interp1d(p_grid, p_2d, s_2d, pad_value=-1.0)

    norm_a = torch.where(aligned_a != -1.0, aligned_a / max_a, -1.0)
    norm_s = torch.where(aligned_s != -1.0, aligned_s / max_s, -1.0)
    norm_p = (p_grid / max_p).view(1, -1)

    return torch.stack([norm_p, norm_a, norm_s], dim=1)

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from {cfg['data']['path']} for normalization bounds...")
    data = torch.load(cfg['data']['path'])
    X_all = data["x"]
    
    # 1. Normalization constants
    GLOBAL_MAX_P = data["p_star_max"]
    GLOBAL_MAX_ALPHA = X_all[:, 1, :][X_all[:, 1, :] != -1.0].max().item()
    GLOBAL_MAX_STIFF = X_all[:, 2, :][X_all[:, 2, :] != -1.0].max().item()
    
    # 2. Initialize Models & Physics
    model = SurfaceInverseModel(cfg).to(device)
    model.load_state_dict(torch.load(cfg['model']['name'], map_location=device))
    model.eval()

    phys_engine = AxisymmetricContactLayer(cfg=cfg).to(device)
    target_gen = TargetGenerator(phys_engine, cfg, device)
    
    n_steps = cfg['data']['n_steps']
    indents = target_gen.indentations
    p_star_grid = torch.linspace(0, GLOBAL_MAX_P, n_steps).to(device)
    w_t = target_gen.t_w

    # 3. CURATE THE TEST CASES
    print("\nPreparing curated test cases...")
    test_cases = {}
    
    # Updated to use 'label' instead of 'category'
    p, a, s, _, _, _ = target_gen.get_custom_sample(idx=5, label="wall")
    test_cases["Wall"] = (p, a, s)
    
    p, a, s, _, _, _ = target_gen.get_custom_sample(idx=5, label="sparse")
    test_cases["Sparse"] = (p, a, s)
    
    p, a, s, _ = target_gen.get_consistent_linear_coulomb()
    test_cases["Linear"] = (p, a, s)

    saved_results = {}

    for name, (target_p, target_a, target_s) in test_cases.items():
        print(f"\n=== Validating & Refining: {name} ===")

        # A. Format for CNN
        nn_input = prepare_nn_input(target_p, target_a, target_s, p_star_grid, GLOBAL_MAX_P, GLOBAL_MAX_ALPHA, GLOBAL_MAX_STIFF)
        
        # B. Zero-Shot Prediction
        with torch.no_grad():
            n_nn_t, h_nn_t = model(nn_input)
            h_nn_t = torch.sort(h_nn_t, dim=1)[0] - torch.sort(h_nn_t, dim=1)[0][:, 0:1]
        
        # C. L-BFGS Refinement (Removed k_steepness and steps)
        n_opt_t, h_opt_t, p_opt_t, a_opt_t, _ = refine_topology(
            target_alpha=target_a, target_p=target_p,
            n_init=n_nn_t, h_init=h_nn_t,
            phys_engine=phys_engine, p_star_grid=p_star_grid,
            t_w=w_t, indentations=indents
        )

        # D. Run Tamaas BEM
        pred_n = n_opt_t.detach().cpu().numpy()[0]
        pred_h = h_opt_t.detach().cpu().numpy()[0]
        
        target_p_np = target_p.cpu().numpy()
        target_a_np = target_a.cpu().numpy()
        a_opt_aligned = np.interp(target_p_np, p_opt_t.detach().cpu().numpy()[0], a_opt_t.detach().cpu().numpy()[0], right=-1.0)

        print(f"  > Running Tamaas on {name} Refined Surface...")
        bem_alphas, bem_surface = None, None
        
        # CRITICAL FIX: Filter out padded values AND microscopic pressures (< 5e-5) 
        # to prevent the "single node" NaN crash in the BEM solver.
        valid_mask = (target_a_np != -1.0) & (target_p_np > 5e-5)
        valid_pressures = target_p_np[valid_mask]
        
        if len(valid_pressures) > 0:
            tamaas_pressure_steps = valid_pressures[::5] # Sample to save time

            try:
                bem_alphas, bem_surface, bem_L = run_tamas_simulation(
                    heights=pred_h, ns=pred_n, 
                    widths=np.ones_like(pred_n) * (2.0 * cfg['physics']['radius']), 
                    target_pressures=tamaas_pressure_steps,
                    L=phys_engine.L, E_star=cfg['physics']['E_star']
                )
                print("  > Tamaas Simulation Complete.")
            except Exception as e:
                print(f"  > Tamaas Failed: {e}")
        else:
            print("  > Skipping Tamaas: No valid macroscopic pressures available.")
            tamaas_pressure_steps = []

        # E. Save Results
        saved_results[f"sample_{name}"] = {
            "pressure_gt": target_p_np,     
            "alpha_gt": target_a_np,           
            "alpha_nn_opt_analytical": a_opt_aligned, 
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