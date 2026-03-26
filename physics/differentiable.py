import torch
import torch.nn as nn
import numpy as np

class AxisymmetricContactLayer(nn.Module):
    def __init__(self, cfg, epsilon=1e-8):
        """
        Args:
            cfg: The full config dictionary to extract R, max_delta, and n_asp.
            epsilon: Small value to prevent division by zero.
        """
        super().__init__()
        self.E = cfg['physics']['E_star']
        self.epsilon = epsilon
        self.sqrt_pi = np.sqrt(np.pi)
        
        # --- GLOBAL DOMAIN CALCULATION (L^2) ---
        self.n_asp = cfg['physics']['n_asperities']
        self.R = cfg['physics']['radius']
        self.max_delta = cfg['physics']['max_delta_ratio'] * self.R
        
        # 1. Worst-case shape: n = 3.0
        n_worst = torch.tensor([3.0])
        k_worst = self.kappa_torch(n_worst).item()
        
        # 2. Worst-case contact radius (a_max)
        # a = (delta/k)^(1/n) * w^((n-1)/n), where w = 2R
        w_worst = 2.0 * self.R
        term1 = self.max_delta / k_worst
        pow_a1 = 1.0 / 3.0
        pow_a2 = 2.0 / 3.0
        a_max = (term1 ** pow_a1) * (w_worst ** pow_a2)
        
        # 3. Safe cell size (4x radius padding)
        self.cell_size = 4.0 * a_max
        
        # 4. Total Macroscopic Domain Area (L^2)
        # N asperities fit into a grid of sqrt(N) x sqrt(N)
        grid_dim = np.sqrt(self.n_asp)
        self.L = grid_dim * self.cell_size
        self.nominal_area = self.L ** 2
        
        print(f"[Physics Engine] Global Domain Locked. L = {self.L:.6f} m, Nominal Area = {self.nominal_area:.2e} m^2")

    def kappa_torch(self, n):
        log_k = torch.lgamma(n / 2.0 + 1.0) - torch.lgamma((n + 1.0) / 2.0)
        return self.sqrt_pi * torch.exp(log_k)

    def forward(self, heights, exponents, widths, indentations):
        """
        Returns:
            nominal_pressure (P): (Batch, M_steps)
            contact_fraction (alpha): (Batch, M_steps)
            dP_dAlpha:            (Batch, M_steps) - The Intensive Stiffness
        """
        h = heights.unsqueeze(2)     
        n = exponents.unsqueeze(2)   
        w = widths.unsqueeze(2)      
        d = indentations.unsqueeze(1)

        delta = d - h 
        is_contact = delta > 0.0

        safe_delta = torch.where(is_contact, delta, torch.ones_like(delta))

        k = self.kappa_torch(n) 
        term1 = safe_delta / k
        
        pow_a1 = 1.0 / n
        pow_a2 = (n - 1.0) / n
        a = torch.pow(term1, pow_a1) * torch.pow(w, pow_a2)

        area_i = np.pi * torch.pow(a, 2)
        factor = self.E * (2.0 * n) / (n + 1.0) * k * torch.pow(w, 1.0 - n)
        load_i = factor * torch.pow(a, n + 1.0)

        # Analytical Derivatives
        dArea_dDelta_i = (2.0 / (n * safe_delta)) * area_i
        dLoad_dDelta_i = ((n + 1.0) / (n * safe_delta)) * load_i

        area_i = area_i * is_contact.float()
        load_i = load_i * is_contact.float()
        dArea_dDelta_i = dArea_dDelta_i * is_contact.float()
        dLoad_dDelta_i = dLoad_dDelta_i * is_contact.float()

        # Extensive Sums
        total_area = area_i.sum(dim=1)               
        total_load = load_i.sum(dim=1)               
        total_dA_dDelta = dArea_dDelta_i.sum(dim=1)  
        total_dF_dDelta = dLoad_dDelta_i.sum(dim=1)  

        # --- NEW: CONVERT TO INTENSIVE PROPERTIES ---
        nominal_pressure = total_load / self.nominal_area
        contact_fraction = total_area / self.nominal_area
        
        # dP/dAlpha is mathematically identical to dF/dA, 
        # but we calculate it directly from the intensive rates to be strictly correct.
        dP_dDelta = total_dF_dDelta / self.nominal_area
        dAlpha_dDelta = total_dA_dDelta / self.nominal_area

        dP_dAlpha = torch.where(
            dAlpha_dDelta > self.epsilon, 
            dP_dDelta / (dAlpha_dDelta + self.epsilon), 
            torch.zeros_like(dP_dDelta)
        )

        return nominal_pressure, contact_fraction, dP_dAlpha