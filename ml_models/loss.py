import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self, w_log=10.0, w_lin=20.0, w_slope=5.0, w_param=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.w_log = w_log
        self.w_lin = w_lin
        self.w_slope = w_slope
        self.w_param = w_param

    def forward(self, pred_curve, target_curve, pred_params=None, target_params=None):
        """
        Args:
            pred_curve: (Batch, 2, Steps) - The reconstructed Load/Area
            target_curve: (Batch, 2, Steps) - The ground truth Load/Area
            pred_params: (Optional) Predicted n, h
            target_params: (Optional) Ground truth n, h
        """

        # Log Loss (Good for small values / initial contact)
        loss_log = self.mse(torch.log1p(pred_curve), torch.log1p(target_curve))

        # Linear Loss (Good for high load / saturation)
        loss_lin = self.mse(pred_curve, target_curve)

        # Slope Loss (Smoothness)
        pred_slope = pred_curve[:, :, 1:] - pred_curve[:, :, :-1]
        targ_slope = target_curve[:, :, 1:] - target_curve[:, :, :-1]
        loss_slope = self.mse(pred_slope, targ_slope)

        # Combined Physics Loss
        total_loss = (loss_log * self.w_log) + \
                     (loss_lin * self.w_lin) + \
                     (loss_slope * self.w_slope)

        # Parameter Loss (Only if targets are provided!)
        if pred_params is not None and target_params is not None:
            loss_param = self.mse(pred_params, target_params)
            total_loss += (loss_param * self.w_param)

        return total_loss
