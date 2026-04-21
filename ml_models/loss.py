import torch
import torch.nn as nn
import torch.nn.functional as F

class CurriculumIntensiveLoss(nn.Module):
    def __init__(self, w_shape=1.0, w_grad=1.0, w_mag=10, max_delta=1e-4):
        super().__init__()
        self.w_shape = w_shape
        self.w_grad = w_grad
        self.w_mag = w_mag
        self.max_delta = max_delta

    def forward(self, pred_alpha_hat, target_alpha_hat, 
                pred_stiff_hat, target_stiff_hat, 
                pred_scalars, target_scalars, 
                pred_params, target_params, lambda_param=0.0):
        
        # ---------------------------------------------------------
        # SHAPE LOSS (Topology & Fingerprint)
        # ---------------------------------------------------------
        # L1 for horizontal shifts in stiffness cliffs, 
        loss_alpha = F.l1_loss(pred_alpha_hat, target_alpha_hat)
        loss_stiff = F.l1_loss(pred_stiff_hat, target_stiff_hat)
        
        # Stabilized Gradient Loss: Ensures the curve 'bends' correctly.
        diff_pred = torch.diff(pred_stiff_hat, dim=1)
        diff_target = torch.diff(target_stiff_hat, dim=1)
        loss_grad = F.l1_loss(diff_pred, diff_target)

        loss_shape_total = (loss_alpha + loss_stiff) * self.w_shape + (loss_grad * self.w_grad)

        # ---------------------------------------------------------
        # MAGNITUDE LOSS (Absolute Physical Scale)
        # ---------------------------------------------------------
        # Extract the absolute maximums [Batch, 2]
        pred_p_max, pred_a_max = pred_scalars[:, 0], pred_scalars[:, 1]
        targ_p_max, targ_a_max = target_scalars[:, 0], target_scalars[:, 1]

        # MSLE: Mean Squared Logarithmic Error
        # Equalizes penalty across orders of magnitude
        log_pred_p = torch.log10(pred_p_max + 1e-12)
        log_targ_p = torch.log10(targ_p_max + 1e-12)
        
        log_pred_a = torch.log10(pred_a_max + 1e-12)
        log_targ_a = torch.log10(targ_a_max + 1e-12)

        loss_mag_p = F.mse_loss(log_pred_p, log_targ_p)
        loss_mag_a = F.mse_loss(log_pred_a, log_targ_a)
        
        loss_mag_total = (loss_mag_p + loss_mag_a) * self.w_mag

        # Total Physics Loss 

        physics_loss = loss_shape_total + loss_mag_total

        # ---------------------------------------------------------
        # PARAMETER LOSS (Curriculum Anchor)
        # ---------------------------------------------------------
        n_asp = pred_params.shape[1] // 2
        pred_n, pred_h = pred_params[:, :n_asp], pred_params[:, n_asp:]
        targ_n, targ_h = target_params[:, :n_asp], target_params[:, n_asp:]
        
        loss_n = F.mse_loss(pred_n / 3.0, targ_n / 3.0)
        loss_h = F.mse_loss(pred_h / self.max_delta, targ_h / self.max_delta)
        param_loss = loss_n + loss_h

        # print(f"Shape: {loss_shape_total.item():.4f} | Mag: {loss_mag_total.item():.4f} | Param: {param_loss.item():.4f} | Lmbda: {lambda_param:.4f}")

        return physics_loss + (lambda_param * param_loss)