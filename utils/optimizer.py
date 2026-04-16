import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils.interpolation import batched_interp1d

def refine_topology(target_alpha, target_p, n_init, h_init, phys_engine, p_star_grid, t_w, indentations, k_start=1e2, k_end=1e5, stages=5, steps_per_stage=5, w_bounds=0.1):
    """
    Refines the initial neural network topography guess using Multi-Stage Homotopy L-BFGS.
    Enforces strict physical bounds: n in [1, 3] and h_0 = 0.
    """
    device = n_init.device

    # 1. INITIALIZE LATENT VARIABLES
    # For n in [1, 3]: Use inverse sigmoid (logit). 
    # We clamp the initial input slightly to prevent logit(0) or logit(1) yielding Infinity.
    n_clamped = torch.clamp((n_init - 1.0) / 2.0, 1e-4, 1.0 - 1e-4)
    n_raw = torch.logit(n_clamped).clone().detach().requires_grad_(True)

    # For h: We only optimize N-1 heights. The anchor is completely removed from the optimizer.
    h_sorted_init, _ = torch.sort(h_init, dim=1)
    h_anchored_init = h_sorted_init - h_sorted_init[:, 0:1]
    h_active_init = h_anchored_init[:, 1:] # Drop the zero anchor
    
    # Inverse softplus for strictly positive active heights
    h_raw = torch.log(torch.exp(h_active_init) - 1.0 + 1e-8).clone().detach().requires_grad_(True)

    # Grids and Masks setup
    t_p_2d = target_p.view(1, -1)
    t_a_2d = target_alpha.view(1, -1)
    t_alpha_aligned = batched_interp1d(p_star_grid, t_p_2d, t_a_2d, pad_value=-1.0)
    target_mask = (t_alpha_aligned != -1.0).float()
    
    k_schedule = np.logspace(np.log10(k_start), np.log10(k_end), num=stages)

    for stage, current_k in enumerate(k_schedule):
        # We optimize the raw latent variables, not the physical ones
        optimizer = optim.LBFGS([n_raw, h_raw], lr=0.5, max_iter=20, line_search_fn='strong_wolfe')
        
        for step in range(steps_per_stage):
            def closure():
                optimizer.zero_grad()
                
                # 2. MAP TO STRICT PHYSICAL BOUNDS
                n_phys = 1.0 + 2.0 * torch.sigmoid(n_raw) # Strictly bounds between 1.0 and 3.0
                h_active = F.softplus(h_raw)              # Strictly > 0
                
                # Reconstruct the full 9-asperity array with an immutable zero anchor
                h_anchor = torch.zeros(h_active.shape[0], 1, device=device)
                h_phys = torch.cat([h_anchor, h_active], dim=1)
                h_phys, _ = torch.sort(h_phys, dim=1)     # Ensure monotonic step ordering

                # Forward physics pass
                p_raw, a_raw, _ = phys_engine(h_phys, n_phys, t_w, indentations, k_steepness=current_k)
                
                p_raw_2d = p_raw.view(1, -1)
                a_raw_2d = a_raw.view(1, -1)

                a_aligned = batched_interp1d(p_star_grid, p_raw_2d, a_raw_2d, pad_value=-1.0)
                pred_mask = (a_aligned != -1.0).float()
                
                overlap_mask = target_mask * pred_mask
                overshoot_mask = (1.0 - target_mask) * pred_mask
                undershoot_mask = target_mask * (1.0 - pred_mask)
                
                mask_weights = overlap_mask + (undershoot_mask * w_bounds) 
                valid_elements = torch.sum(mask_weights) + 1e-8 

                squared_err = torch.pow(a_aligned - t_alpha_aligned, 2)
                loss = torch.sum(squared_err * mask_weights) / valid_elements
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
            except Exception as e:
                print(f"    [Warning] Stage {stage+1} (k={current_k:.0f}) Optimization step failed: {e}")
                break

    # Final mapping to extract the exact optimized physical parameters
    with torch.no_grad():
        n_final = 1.0 + 2.0 * torch.sigmoid(n_raw)
        h_active_final = F.softplus(h_raw)
        
        h_anchor_final = torch.zeros(h_active_final.shape[0], 1, device=device)
        h_final = torch.cat([h_anchor_final, h_active_final], dim=1)
        h_final, _ = torch.sort(h_final, dim=1)
        
        p_final, alpha_final, s_final = phys_engine(h_final, n_final, t_w, indentations, k_steepness=k_end)

    return n_final, h_final, p_final, alpha_final, s_final