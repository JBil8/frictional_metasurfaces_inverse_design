import torch
import torch.optim as optim
import numpy as np
from utils.interpolation import batched_interp1d

def refine_topology(target_alpha, target_p, n_init, h_init, phys_engine, p_star_grid, t_w, indentations, k_start=1e2, k_end=1e5, stages=5, steps_per_stage=5, w_bounds=0.1):
    """
    Refines the initial neural network topography guess using Multi-Stage Homotopy L-BFGS.
    """

    # Detach and require grad for the parameters we want to optimize
    n_opt = n_init.clone().detach().requires_grad_(True)
    h_opt = h_init.clone().detach().requires_grad_(True)

    # Force 2D shape (1, Steps)
    t_p_2d = target_p.view(1, -1)
    t_a_2d = target_alpha.view(1, -1)

    # Align target to grid once
    t_alpha_aligned = batched_interp1d(p_star_grid, t_p_2d, t_a_2d, pad_value=-1.0)
    target_mask = (t_alpha_aligned != -1.0).float()
    
    # Create the logarithmic annealing schedule for k
    k_schedule = np.logspace(np.log10(k_start), np.log10(k_end), num=stages)

    for stage, current_k in enumerate(k_schedule):
        # We must re-instantiate the optimizer for each stage.
        # This flushes the L-BFGS Hessian memory, which is invalid because the loss landscape 
        # has just changed due to the new current_k value.
        optimizer = optim.LBFGS([n_opt, h_opt], lr=0.5, max_iter=20, line_search_fn='strong_wolfe')
        
        for step in range(steps_per_stage):
            def closure():
                optimizer.zero_grad()
                
                h_sorted, _ = torch.sort(h_opt, dim=1)
                h_sorted = h_sorted - h_sorted[:, 0:1]

                # Forward physics pass using the CURRENT stage's steepness
                p_raw, a_raw, _ = phys_engine(h_sorted, n_opt, t_w, indentations, k_steepness=current_k)
                
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

    # Final forward pass to lock in the optimized variables at the STRICT physics limit
    with torch.no_grad():
        h_final, _ = torch.sort(h_opt, dim=1)
        h_final = h_final - h_final[:, 0:1]
        p_final, alpha_final, s_final = phys_engine(h_final, n_opt, t_w, indentations, k_steepness=k_end)

    return n_opt, h_final, p_final, alpha_final, s_final