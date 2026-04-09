import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AxisymmetricContactLayer(nn.Module):
    def __init__(self, cfg, epsilon=1e-8):
        super().__init__()
        self.E = cfg['physics']['E_star']
        self.epsilon = epsilon
        self.sqrt_pi = np.sqrt(np.pi)
        
        self.n_asp = cfg['physics']['n_asperities']
        self.R = cfg['physics']['radius']
        self.max_delta = cfg['physics']['max_delta_ratio'] * self.R
        
        # Domain Calculation
        n_worst = torch.tensor([3.0])
        k_worst = self.kappa_torch(n_worst).item()
        w_worst = 2.0 * self.R
        a_max = ((self.max_delta / k_worst) ** (1.0 / 3.0)) * (w_worst ** (2.0 / 3.0))
        
        self.cell_size = 4.0 * a_max
        self.nominal_area = (np.sqrt(self.n_asp) * self.cell_size) ** 2
        print(f"[Physics Engine] Global Domain Locked. Nominal Area = {self.nominal_area:.2e} m^2")

    def kappa_torch(self, n):
        log_k = torch.lgamma(n / 2.0 + 1.0) - torch.lgamma((n + 1.0) / 2.0)
        return self.sqrt_pi * torch.exp(log_k)

    def forward(self, heights, exponents, widths, indentations, k_steepness=1e6):
        h = heights.unsqueeze(2)     
        n = exponents.unsqueeze(2)   
        w = widths.unsqueeze(2)      
        d = indentations.unsqueeze(1)

        # 1. SOFTPLUS HOMOTOPY (Fixes the ghost gradient)
        # This naturally curves smoothly to 0 for negative deltas, avoiding the need for manual clamping logic
        smooth_delta = F.softplus(d - h, beta=k_steepness)
        
        # 2. Prevent Inf/NaN in derivatives when delta is perfectly zero
        safe_delta = torch.clamp(smooth_delta, min=1e-8)

        k = self.kappa_torch(n) 
        
        pow_a1 = 1.0 / n
        pow_a2 = (n - 1.0) / n
        a = torch.pow(safe_delta / k, pow_a1) * torch.pow(w, pow_a2)

        # 3. Raw Extensive Properties
        area_i = np.pi * torch.pow(a, 2.0)
        factor = self.E * (2.0 * n) / (n + 1.0) * k * torch.pow(w, 1.0 - n)
        load_i = factor * torch.pow(a, n + 1.0)

        # 4. Chain Rule Derivatives 
        # d(Area)/d(d) = d(Area)/d(smooth_delta) * d(smooth_delta)/d(d)
        # The derivative of softplus is exactly the sigmoid
        contact_weight = torch.sigmoid(k_steepness * (d - h))
        
        dArea_dDelta_i = ((2.0 / (n * safe_delta)) * area_i) * contact_weight
        dLoad_dDelta_i = (((n + 1.0) / (n * safe_delta)) * load_i) * contact_weight

        # 5. Intensive Properties
        nominal_pressure = load_i.sum(dim=1) / self.nominal_area
        contact_fraction = area_i.sum(dim=1) / self.nominal_area
        
        dP_dDelta = dLoad_dDelta_i.sum(dim=1) / self.nominal_area
        dAlpha_dDelta = dArea_dDelta_i.sum(dim=1) / self.nominal_area

        dP_dAlpha = torch.where(
            dAlpha_dDelta > self.epsilon, 
            dP_dDelta / (dAlpha_dDelta + self.epsilon), 
            torch.zeros_like(dP_dDelta)
        )

        return nominal_pressure, contact_fraction, dP_dAlpha