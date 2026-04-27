import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils.interpolation import batched_interp1d

def refine_topology(target_alpha, target_p, n_init, h_init, phys_engine, p_star_grid, t_w,
                    indentations, gamma_min=1.8, gamma_max=4.0, k_start=1e3, k_end=1e5, 
                    stages=5, steps_per_stage=5, w_bounds=0.1, lock_n=False):
    """
    Refines the initial topography guess using Multi-Stage Homotopy L-BFGS.
    If lock_n=True, it strictly enforces Hertzian spherical asperities (n=2).
    """
    device = n_init.device

    # --- H Initialization (Always Optimized) ---
    h_sorted_init, _ = torch.sort(h_init, dim=1)
    h_anchored_init = h_sorted_init - h_sorted_init[:, 0:1]
    h_active_init = h_anchored_init[:, 1:] 
    h_raw = torch.log(torch.exp(h_active_init) - 1.0 + 1e-8).clone().detach().requires_grad_(True)

    # --- N Initialization (Conditional) ---
    if not lock_n:
        # INVERSE MAPPING: Dynamically invert the bounds to find the starting logit
        n_clamped = torch.clamp((n_init - gamma_min) / (gamma_max - gamma_min), 1e-4, 1.0 - 1e-4)
        n_raw = torch.logit(n_clamped).clone().detach().requires_grad_(True)
        opt_params = [n_raw, h_raw]
    else:
        # Remove N from the optimizer entirely
        opt_params = [h_raw] 

    # Grids and Masks
    t_p_2d = target_p.view(1, -1)
    t_a_2d = target_alpha.view(1, -1)
    t_alpha_aligned = batched_interp1d(p_star_grid, t_p_2d, t_a_2d, pad_value=-1.0)
    target_mask = (t_alpha_aligned != -1.0).float()
    
    k_schedule = np.logspace(np.log10(k_start), np.log10(k_end), num=stages)

    for stage, current_k in enumerate(k_schedule):
        optimizer = optim.LBFGS(opt_params, lr=0.5, max_iter=20, line_search_fn='strong_wolfe')
        
        for step in range(steps_per_stage):
            def closure():
                optimizer.zero_grad()
                
                # --- Enforce Bounds ---
                h_active = F.softplus(h_raw)
                h_anchor = torch.zeros(h_active.shape[0], 1, device=device)
                h_phys = torch.cat([h_anchor, h_active], dim=1)
                h_phys, _ = torch.sort(h_phys, dim=1)

                if not lock_n:
                    # FORWARD MAPPING: Dynamically scale the sigmoid output to the config bounds
                    n_phys = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(n_raw)
                else:
                    n_phys = torch.full_like(n_init, 2.0, device=device) # Strictly Hertzian

                # Forward physics pass
                p_raw, a_raw, _ = phys_engine(h_phys, n_phys, t_w, indentations, k_steepness=current_k)
                
                a_aligned = batched_interp1d(p_star_grid, p_raw.view(1, -1), a_raw.view(1, -1), pad_value=-1.0)
                pred_mask = (a_aligned != -1.0).float()
                
                overlap_mask = target_mask * pred_mask
                overshoot_mask = (1.0 - target_mask) * pred_mask
                undershoot_mask = target_mask * (1.0 - pred_mask)
                
                mask_weights = overlap_mask + (undershoot_mask * w_bounds) 
                valid_elements = torch.sum(mask_weights) + 1e-8 

                loss = torch.sum(torch.pow(a_aligned - t_alpha_aligned, 2) * mask_weights) / valid_elements
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
            except Exception as e:
                break

    # Final mapping
    with torch.no_grad():
        h_active_final = F.softplus(h_raw)
        h_anchor_final = torch.zeros(h_active_final.shape[0], 1, device=device)
        h_final = torch.cat([h_anchor_final, h_active_final], dim=1)
        h_final, _ = torch.sort(h_final, dim=1)

        if not lock_n:
            n_final = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(n_raw)
        else:
            n_final = torch.full_like(n_init, 2.0, device=device)
            
        p_final, alpha_final, s_final = phys_engine(h_final, n_final, t_w, indentations, k_steepness=k_end)

    return n_final, h_final, p_final, alpha_final, s_final