import torch
import torch.nn as nn

class CurriculumIntensiveLoss(nn.Module):
    def __init__(self, w_stiff=1.0, w_pressure=2.0, w_bounds=0.1, max_delta=1e-4):
        super().__init__()
        self.w_stiff = w_stiff
        self.w_alpha = w_pressure 
        self.w_bounds = w_bounds 
        self.max_delta = max_delta

    def forward(self, pred_stiff, target_stiff, pred_alpha, target_alpha, pred_params, target_params, lambda_param=0.0):
        # ---------------------------------------------------------
        # 1. SPLIT MASK WEIGHTING
        # ---------------------------------------------------------
        pred_mask = (pred_stiff != -1.0).float()
        target_mask = (target_stiff != -1.0).float()

        # Zone 1: Pure Physics (Both curves exist here)
        overlap_mask = target_mask * pred_mask

        # Zone 2: Overshoot (Target bottomed out, Network kept going)
        overshoot_mask = (1.0 - target_mask) * pred_mask

        # Zone 3: Undershoot (Target is physical, Network bottomed out early)
        undershoot_mask = target_mask * (1.0 - pred_mask)

        # Build the composite weight map
        # Physics gets 1.0 multiplier. Boundary violations get w_bounds (e.g., 0.1) multiplier.
        mask_weights = overlap_mask +  (undershoot_mask * self.w_bounds)
        valid_elements = torch.sum(mask_weights) + 1e-8 

        # ---------------------------------------------------------
        # 2. LINEAR MASKED CURVE LOSS
        # ---------------------------------------------------------
        # Standard MSE, but scaled by our custom zone weights
        squared_err_stiff = torch.pow(pred_stiff - target_stiff, 2)
        loss_shape = torch.sum(squared_err_stiff * mask_weights) / valid_elements

        squared_err_alpha = torch.pow(pred_alpha - target_alpha, 2)
        loss_alpha = torch.sum(squared_err_alpha * mask_weights) / valid_elements

        # ---------------------------------------------------------
        # 3. OVERLAP-ONLY GRADIENT (SLOPE) LOSS
        # ---------------------------------------------------------
        # Calculate slopes only where BOTH curves exist physically
        diff_pred = torch.diff(pred_stiff, dim=2)
        diff_target = torch.diff(target_stiff, dim=2)

        # Shift the overlap mask to match the diff dimensions
        diff_mask = overlap_mask[:, :, :-1] * overlap_mask[:, :, 1:]
        diff_elements = torch.sum(diff_mask) + 1e-8

        # L1 Loss on the physical slopes
        loss_grad = torch.sum(torch.abs(diff_pred - diff_target) * diff_mask) / diff_elements

        # Total Physics Loss
        physics_loss = ((loss_shape + loss_grad) * self.w_stiff) + (loss_alpha * self.w_alpha)

        # ---------------------------------------------------------
        # 4. PARAMETER LOSS 
        # ---------------------------------------------------------
        n_asp = pred_params.shape[1] // 2
        pred_n, pred_h = pred_params[:, :n_asp], pred_params[:, n_asp:]
        targ_n, targ_h = target_params[:, :n_asp], target_params[:, n_asp:]
        
        loss_n = nn.MSELoss()(pred_n / 3.0, targ_n / 3.0)
        loss_h = nn.MSELoss()(pred_h / self.max_delta, targ_h / self.max_delta)
        param_loss = loss_n + loss_h

        return physics_loss + (lambda_param * param_loss)