import torch
import torch.nn as nn

class StiffnessLoss(nn.Module):
    def __init__(self, w_stiff=1.0, w_grad=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() # <--- Use L1 for the discontinuous gradients
        self.w_stiff = w_stiff
        self.w_grad = w_grad 

    def forward(self, pred_curve, target_curve):
        # Base Topology Loss (Handles the smooth, continuous regions well)
        loss_stiff = self.mse(pred_curve, target_curve)

        # Gradient Loss (Forces the Cliffs to align without exploding the gradients)
        pred_diff = torch.diff(pred_curve, dim=2)
        target_diff = torch.diff(target_curve, dim=2)
        loss_grad = self.l1(pred_diff, target_diff)

        return (loss_stiff * self.w_stiff) + (loss_grad * self.w_grad)