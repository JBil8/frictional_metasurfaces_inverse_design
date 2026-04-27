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
from ml_models.model_mlp import SurfaceInverseModel # Assuming you renamed this from the CNN
from physics.differentiable import AxisymmetricContactLayer
from physics.tamas_solution import run_tamas_simulation
from utils.optimizer import refine_topology
from utils.interpolation import batched_interp1d
from utils.seeding import set_seed
from validation.validator import UnifiedValidator

try:
    from validation.targets import TargetGenerator
except ImportError:
    from targets import TargetGenerator

def prepare_nn_input(t_p, t_a, t_s, n_steps=512, device='cuda'):
    """
    Decoupled Architecture Input:
    1. Extracts local absolute maximums (Scalars)
    2. Normalizes curves locally into [0, 1] domain (Arrays)
    3. Interpolates onto a strict uniform p_hat_grid
    """
    # 1. Extract local maximums (Ensuring no division by zero)
    p_max = torch.clamp(t_p[:, -1:], min=1e-12)
    a_max = torch.clamp(t_a[:, -1:], min=1e-12)
    
    # 2. Locally Normalize (Broadcasting handles the division)
    p_hat = t_p / p_max
    a_hat = t_a / a_max
    s_hat = t_s * (a_max / p_max)
    
    # 3. Interpolate onto the strict [0, 1] network grid
    p_hat_grid = torch.linspace(0, 1.0, n_steps, device=device)
    
    a_interp = batched_interp1d(p_hat_grid, p_hat, a_hat, pad_value=1.0)
    s_interp = batched_interp1d(p_hat_grid, p_hat, s_hat, pad_value=0.0)
    
    # 4. Format exactly as the MLP expects
    x_arr = torch.stack([a_interp, s_interp], dim=1)          # Shape: [Batch, 2, 512]
    x_scal_log = torch.log10(torch.cat([p_max, a_max], dim=1)) # Shape: [Batch, 2]
    
    return x_arr, x_scal_log

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = cfg['data']['n_steps']
    set_seed()
    
    # 1. Initialize Models & Physics
    model = SurfaceInverseModel(cfg).to(device)
    model.load_state_dict(torch.load(cfg['model']['name'], map_location=device))
    model.eval()

    phys_engine = AxisymmetricContactLayer(cfg=cfg).to(device)
    target_gen = TargetGenerator(phys_engine, cfg, device)
    
    indents = target_gen.indentations
    w_t = target_gen.t_w

    # 2. CURATE THE TEST CASES
    print("\nPreparing curated test cases for all manifolds...")
    
    test_cases = {}
    
    # Instantiate your existing validator (ensure it uses the same seed=42 as training!)
    val = UnifiedValidator("config.yaml") 
    categorized_indices = val.get_test_set_indices_by_category()

    for cat, indices in categorized_indices.items():
        if not indices:
            continue
            
        try:
            # 1. Pick one random unseen sample from this category's test pool
            idx = np.random.choice(indices) + 1 
            
            # 2. Fetch Ground Truth directly from the generator
            p, a, s, _, _, _ = target_gen.get_custom_sample(idx, cat)
            test_cases[cat] = (p, a, s)
            
        except Exception as e:
            print(f"Skipping {cat}: {e}")
    try:
        p, a, s, _ = target_gen.get_consistent_linear_coulomb()
        test_cases["linear"] = (p, a, s)
    except:
        pass

    saved_results = {}

    for name, (target_p, target_a, target_s) in test_cases.items():
        print(f"\n=== Validating & Refining: {name.upper()} ===")

        # A. Format for New MLP Surrogate
        x_arr, x_scal_log = prepare_nn_input(target_p, target_a, target_s, n_steps, device)
        
        # B. Zero-Shot Prediction (Decoupled forward pass)
        with torch.no_grad():
            n_nn_t, h_nn_t = model(x_arr, x_scal_log)
            # Anchor heights to zero
            h_nn_t = torch.sort(h_nn_t, dim=1)[0] - torch.sort(h_nn_t, dim=1)[0][:, 0:1]
        
        # C. L-BFGS Refinement 
        # Create a local absolute pressure grid for the optimizer to use
        local_p_max = target_p.max().item()
        opt_eval_grid = torch.linspace(0, local_p_max, n_steps, device=device)
        
        n_opt_t, h_opt_t, p_opt_t, a_opt_t, _ = refine_topology(
            target_alpha=target_a, target_p=target_p,
            n_init=n_nn_t, h_init=h_nn_t,
            phys_engine=phys_engine, p_star_grid=opt_eval_grid,
            t_w=w_t, indentations=indents
        )

        # D. Run Tamaas BEM
        pred_n = n_opt_t.detach().cpu().numpy()[0]
        pred_h = h_opt_t.detach().cpu().numpy()[0]
        
        target_p_np = target_p.cpu().numpy()
        target_a_np = target_a.cpu().numpy()
        
        # Realign the L-BFGS output for the plot
        a_opt_aligned = np.interp(target_p_np, p_opt_t.detach().cpu().numpy()[0], a_opt_t.detach().cpu().numpy()[0], right=-1.0)

        print(f"  > Running Tamaas on {name.upper()} Refined Surface...")
        bem_alphas, bem_surface, bem_pressure_field = None, None, None
        bem_L = phys_engine.L
        
        # Filter out padded values AND microscopic pressures (< 5e-5) 
        valid_mask = (target_a_np != -1.0) & (target_p_np > 5e-5)
        valid_pressures = target_p_np[valid_mask]
        
        if len(valid_pressures) > 0:
            tamaas_pressure_steps = valid_pressures[::10] # Downsample to save time

            try:
                # Expecting the 4 updated variables from run_tamas_simulation
                bem_alphas, bem_surface, bem_pressure_field, bem_L = run_tamas_simulation(
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
            "pressure_field_bem": bem_pressure_field,
            "L_bem": bem_L                                             
        }  

    os.makedirs("./data", exist_ok=True)
    np.savez("./data/paper_validation_data.npz", **saved_results)
    print("\nAll validation data saved to 'data/paper_validation_data.npz'")

if __name__ == "__main__":
    main()