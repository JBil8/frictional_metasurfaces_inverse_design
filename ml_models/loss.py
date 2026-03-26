import torch
import torch.nn as nn

class CurriculumIntensiveLoss(nn.Module):
    def __init__(self, w_stiff=1.0, w_pressure=2.0, max_delta=1e-4):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.w_stiff = w_stiff
        self.w_pressure = w_pressure
        self.max_delta = max_delta

    def forward(self, pred_stiff, target_stiff, pred_pressure, target_pressure, pred_params, target_params, lambda_param):
        # 1. Physics Loss 
        loss_shape = self.mse(pred_stiff, target_stiff)
        loss_grad = self.l1(torch.diff(pred_stiff, dim=2), torch.diff(target_stiff, dim=2))
        loss_p = self.mse(pred_pressure, target_pressure)
        
        physics_loss = ((loss_shape + loss_grad) * self.w_stiff) + (loss_p * self.w_pressure)

        # 2. Normalized Parameter Loss
        n_asp = pred_params.shape[1] // 2
        pred_n, pred_h = pred_params[:, :n_asp], pred_params[:, n_asp:]
        targ_n, targ_h = target_params[:, :n_asp], target_params[:, n_asp:]
        
        # Scale both to roughly [0, 1] so the optimizer cares about both equally
        loss_n = self.mse(pred_n / 3.0, targ_n / 3.0)
        loss_h = self.mse(pred_h / self.max_delta, targ_h / self.max_delta)
        
        param_loss = loss_n + loss_h

        return physics_loss + (lambda_param * param_loss)