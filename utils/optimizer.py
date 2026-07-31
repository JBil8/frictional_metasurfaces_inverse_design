import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils.interpolation import batched_interp1d


def refine_topology(target_p, target_alpha, target_stiff, n_init, h_init, phys_engine,
                    p_hat_grid, criterion, t_w, indentations,
                    gamma_min=1.8, gamma_max=4.0, k_start=1e3, k_end=1e5,
                    stages=5, steps_per_stage=5, lock_n=False):
    """
    Refines the initial topography guess using Multi-Stage Homotopy L-BFGS.
    Optimizes using the exact CurriculumIntensiveLoss (lambda=0).
    """
    device = n_init.device

    # --- 1. PREPARE NORMALIZED TARGETS ONCE ---
    with torch.no_grad():
        t_p_max = torch.clamp(target_p[:, -1:], min=1e-12)
        t_a_max = torch.clamp(target_alpha[:, -1:], min=1e-12)
        target_scalars = torch.cat([t_p_max, t_a_max], dim=1)

        t_p_hat = target_p / t_p_max
        t_a_hat = target_alpha / t_a_max
        t_s_hat = target_stiff * (t_a_max / t_p_max)

        target_alpha_hat = batched_interp1d(
            p_hat_grid, t_p_hat, t_a_hat, pad_value=1.0)
        target_stiff_hat = batched_interp1d(
            p_hat_grid, t_p_hat, t_s_hat, pad_value=0.0)

        # Dummy parameter tensors (ignored since lambda_param = 0)
        dummy_params = torch.zeros(1, 2 * h_init.shape[1]).to(device)

    # --- 2. H Initialization (Always Optimized) ---
    h_sorted_init, _ = torch.sort(h_init, dim=1)
    h_anchored_init = h_sorted_init - h_sorted_init[:, 0:1]
    h_active_init = h_anchored_init[:, 1:]
    h_raw = torch.log(torch.exp(h_active_init) - 1.0 +
                      1e-8).clone().detach().requires_grad_(True)

    # --- 3. N Initialization (Conditional) ---
    if not lock_n:
        n_clamped = torch.clamp((n_init - gamma_min) /
                                (gamma_max - gamma_min), 1e-4, 1.0 - 1e-4)
        n_raw = torch.logit(n_clamped).clone().detach().requires_grad_(True)
        opt_params = [n_raw, h_raw]
    else:
        opt_params = [h_raw]

    k_schedule = np.logspace(np.log10(k_start), np.log10(k_end), num=stages)

    for stage, current_k in enumerate(k_schedule):
        optimizer = optim.LBFGS(
            opt_params, lr=0.5, max_iter=20, line_search_fn='strong_wolfe')

        for step in range(steps_per_stage):
            def closure():
                optimizer.zero_grad()

                # --- Enforce Bounds ---
                h_active = F.softplus(h_raw)
                h_anchor = torch.zeros(h_active.shape[0], 1, device=device)
                h_phys = torch.cat([h_anchor, h_active], dim=1)
                h_phys, _ = torch.sort(h_phys, dim=1)

                if not lock_n:
                    n_phys = gamma_min + \
                        (gamma_max - gamma_min) * torch.sigmoid(n_raw)
                else:
                    n_phys = torch.full_like(n_init, 2.0, device=device)

                # --- Forward physics pass ---
                p_raw, a_raw, s_raw = phys_engine(
                    h_phys, n_phys, t_w, indentations, k_steepness=current_k)

                # --- Normalize Predictions ---
                pred_p_max = torch.clamp(p_raw[:, -1:], min=1e-12)
                pred_a_max = torch.clamp(a_raw[:, -1:], min=1e-12)
                pred_scalars = torch.cat([pred_p_max, pred_a_max], dim=1)

                p_hat_pred = p_raw / pred_p_max
                a_hat_pred = a_raw / pred_a_max
                s_hat_pred = s_raw * (pred_a_max / pred_p_max)

                # --- Interpolate to Hat Grid ---
                pred_alpha_interp = batched_interp1d(
                    p_hat_grid, p_hat_pred, a_hat_pred, pad_value=1.0)
                pred_stiff_interp = batched_interp1d(
                    p_hat_grid, p_hat_pred, s_hat_pred, pad_value=0.0)

                # --- Compute Full Loss ---
                loss = criterion(
                    pred_alpha_hat=pred_alpha_interp,
                    target_alpha_hat=target_alpha_hat,
                    pred_stiff_hat=pred_stiff_interp,
                    target_stiff_hat=target_stiff_hat,
                    pred_scalars=pred_scalars,
                    target_scalars=target_scalars,
                    pred_params=dummy_params,
                    target_params=dummy_params,
                    lambda_param=0.0
                )

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
            n_final = gamma_min + (gamma_max - gamma_min) * \
                torch.sigmoid(n_raw)
        else:
            n_final = torch.full_like(n_init, 2.0, device=device)

        p_final, alpha_final, s_final = phys_engine(
            h_final, n_final, t_w, indentations, k_steepness=k_end)

    return n_final, h_final, p_final, alpha_final, s_final
