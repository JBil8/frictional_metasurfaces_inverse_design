import torch
import torch.nn as nn
import numpy as np

class AxisymmetricContactLayer(nn.Module):
    def __init__(self, E_star=1.0, epsilon=1e-8):
        """
        Args:
            E_star: Effective Young's Modulus (scalar).
            epsilon: Small value to avoid division by zero or NaN gradients.
        """
        super().__init__()
        # Register_buffer so E moves to GPU automatically with model.to(device)
        self.register_buffer('E', torch.tensor(float(E_star)))
        self.register_buffer('sqrt_pi', torch.tensor(np.sqrt(np.pi), dtype=torch.float32))
        self.epsilon = epsilon

    def kappa_torch(self, n):
        """
        Differentiable calculation of Kappa using Log-Gamma.
        kappa = sqrt(pi) * gamma(n/2 + 1) / gamma((n+1)/2)
        """
        log_k = torch.lgamma(n / 2.0 + 1.0) - torch.lgamma((n + 1.0) / 2.0)
        return self.sqrt_pi * torch.exp(log_k)

    def forward(self, heights, exponents, widths, indentations):
        """
        Computes Load, Area, and the Marginal Friction (dF/dA) 
        for a surface given the geometry and indentation steps.
        
        Returns:
            total_load: (Batch, M_steps)
            total_area: (Batch, M_steps)
            dF_dA:      (Batch, M_steps) - The Stiffness/Marginal Friction
        """
        h = heights.unsqueeze(2)     # [B, N, 1]
        n = exponents.unsqueeze(2)   # [B, N, 1]
        w = widths.unsqueeze(2)      # [B, N, 1]
        d = indentations.unsqueeze(1)# [B, 1, M]

        # Calculate Overlap (Delta)
        delta = d - h 
        is_contact = delta > 0.0

        # STABILITY TRICK: Prevent NaN when delta <= 0
        safe_delta = torch.where(is_contact, delta, torch.ones_like(delta))

        # 1. Geometry Calculations
        k = self.kappa_torch(n) 
        term1 = safe_delta / k
        
        pow_a1 = 1.0 / n
        pow_a2 = (n - 1.0) / n
        a = torch.pow(term1, pow_a1) * torch.pow(w, pow_a2)

        # 2. Base Load and Area
        area_i = np.pi * torch.pow(a, 2)
        factor = self.E * (2.0 * n) / (n + 1.0) * k * torch.pow(w, 1.0 - n)
        load_i = factor * torch.pow(a, n + 1.0)

        # 3. Analytical Derivatives (dF/d_delta and dA/d_delta)
        # Using the power-rule identity: d(x^p)/dx = p/x * x^p
        dArea_dDelta_i = (2.0 / (n * safe_delta)) * area_i
        dLoad_dDelta_i = ((n + 1.0) / (n * safe_delta)) * load_i

        # 4. Apply Mask (Zero out non-contact)
        area_i = area_i * is_contact.float()
        load_i = load_i * is_contact.float()
        dArea_dDelta_i = dArea_dDelta_i * is_contact.float()
        dLoad_dDelta_i = dLoad_dDelta_i * is_contact.float()

        # 5. Sum over asperities to get global macroscopic curves
        total_area = area_i.sum(dim=1)               # [B, M]
        total_load = load_i.sum(dim=1)               # [B, M]
        total_dA_dDelta = dArea_dDelta_i.sum(dim=1)  # [B, M]
        total_dF_dDelta = dLoad_dDelta_i.sum(dim=1)  # [B, M]

        # 6. Chain Rule for the Target Objective: dF/dA = (dF/dDelta) / (dA/dDelta)
        # STABILITY TRICK 2: If total_dA_dDelta is 0 (no asperities touching), 
        # dF/dA is undefined. We force it to 0.0 to avoid NaN losses.
        dF_dA = torch.where(
            total_dA_dDelta > self.epsilon, 
            total_dF_dDelta / (total_dA_dDelta + self.epsilon), 
            torch.zeros_like(total_dF_dDelta)
        )

        return total_load, total_area, dF_dA