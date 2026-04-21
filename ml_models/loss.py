import torch
import torch.nn as nn
import torch.nn.functional as F

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

        overlap_mask = target_mask * pred_mask
        undershoot_mask = target_mask * (1.0 - pred_mask)
        
        # Unpenalized overshoot acts as an exile zone
        mask_weights = overlap_mask + (undershoot_mask * self.w_bounds)
        valid_elements = torch.sum(mask_weights) + 1e-8 

        # ---------------------------------------------------------
        # 2. HUBER MASKED CURVE LOSS (Balances topographies)
        # ---------------------------------------------------------
        # Smooth L1 prevents massive stiffness jumps (Walls) from dominating low-load regions (Sparse)
        err_stiff = F.smooth_l1_loss(pred_stiff, target_stiff, reduction='none', beta=0.05)
        loss_shape = torch.sum(err_stiff * mask_weights) / valid_elements

        err_alpha = F.smooth_l1_loss(pred_alpha, target_alpha, reduction='none', beta=0.05)
        loss_alpha = torch.sum(err_alpha * mask_weights) / valid_elements

        # ---------------------------------------------------------
        # 3. STABILIZED GRADIENT LOSS
        # ---------------------------------------------------------
        # Calculate slopes only where BOTH curves exist physically
        diff_pred = torch.diff(pred_stiff, dim=2)
        diff_target = torch.diff(target_stiff, dim=2)

        diff_mask = overlap_mask[:, :, :-1] * overlap_mask[:, :, 1:]
        diff_elements = torch.sum(diff_mask) + 1e-8

        # We must use Smooth L1 here as well because this is a second derivative!
        err_grad = F.smooth_l1_loss(diff_pred, diff_target, reduction='none', beta=0.01)
        loss_grad = torch.sum(err_grad * diff_mask) / diff_elements

        # Total Physics Loss
        physics_loss = ((loss_shape + loss_grad) * self.w_stiff) + (loss_alpha * self.w_alpha)

        # ---------------------------------------------------------
        # 4. PARAMETER LOSS 
        # ---------------------------------------------------------
        n_asp = pred_params.shape[1] // 2
        pred_n, pred_h = pred_params[:, :n_asp], pred_params[:, n_asp:]
        targ_n, targ_h = target_params[:, :n_asp], target_params[:, n_asp:]
        
        loss_n = F.mse_loss(pred_n / 3.0, targ_n / 3.0)
        loss_h = F.mse_loss(pred_h / self.max_delta, targ_h / self.max_delta)
        param_loss = loss_n + loss_h

        print(f"Physics Loss: {physics_loss.item():.4f}, Param Loss: {param_loss.item():.4f}, Lambda: {lambda_param:.4f}")

        return physics_loss + (lambda_param * param_loss)