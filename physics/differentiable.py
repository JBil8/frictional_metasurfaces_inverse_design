import torch
import torch.nn as nn
import numpy as np

class AxisymmetricContactLayer(nn.Module):
    def __init__(self, E_star=1.0, epsilon=1e-16):
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
        # lgamma is stable. 
        # We ensure n is float32 to match model precision
        log_k = torch.lgamma(n / 2.0 + 1.0) - torch.lgamma((n + 1.0) / 2.0)
        return self.sqrt_pi * torch.exp(log_k)

    def forward(self, heights, exponents, widths, indentations):
        """
        Computes Load and Area for a surface given the geometry and indentation steps.
        
        Args:
            heights (h):    (Batch, N_asperities) - Height offsets (z_i)
            exponents (n):  (Batch, N_asperities) - Shape exponent (n=2 is Hertz)
            widths (w):     (Batch, N_asperities) - Characteristic width (Diameter for Hertz)
            indentations (d): (Batch, M_steps)    - Global indentation depth
            
        Returns:
            total_load: (Batch, M_steps)
            total_area: (Batch, M_steps)
        """
        # Expand dimensions for broadcasting (Batch, N_asp, M_steps)
        h = heights.unsqueeze(2)     # [B, N, 1]
        n = exponents.unsqueeze(2)   # [B, N, 1]
        w = widths.unsqueeze(2)      # [B, N, 1]
        d = indentations.unsqueeze(1)# [B, 1, M]

        # Calculate Overlap (Delta)
        # Overlap = Global Indentation - Asperity Height Offset
        # (If h=0 is the tallest asperity, it touches first).
        delta = d - h 
        
        # Create a Contact Mask (Where overlap > 0)
        # We use this mask to avoid computing powers of negative numbers or zero
        is_contact = delta > 0.0

        # --- NUMERICAL STABILITY TRICK ---
        # We replace delta <= 0 with 1.0 temporarily to calculate power without NaN gradients.
        # We will zero out the result later using the mask.
        safe_delta = torch.where(is_contact, delta, torch.ones_like(delta))

        # Geometry Calculations
        k = self.kappa_torch(n) 
        
        # Calculate Contact Radius (a)
        # a = (delta/k)^(1/n) * w^((n-1)/n)
        # Note: For Hertz (n=2), power is 0.5. Derivative at 0 is Inf.
        # safe_delta handles the value, but gradients can still be unstable if exactly 0.
        term1 = safe_delta / k
        
        # Exponents for radii
        pow_a1 = 1.0 / n
        pow_a2 = (n - 1.0) / n
        
        a = torch.pow(term1, pow_a1) * torch.pow(w, pow_a2)

        # Calculate Individual Load and Area
        area_i = np.pi * torch.pow(a, 2)
        
        # Load factor
        factor = self.E * (2.0 * n) / (n + 1.0) * k * torch.pow(w, 1.0 - n)
        load_i = factor * torch.pow(a, n + 1.0)

        # 6. Apply Mask (Zero out non-contact)
        area_i = area_i * is_contact.float()
        load_i = load_i * is_contact.float()

        # 7. Sum over asperities to get global curve
        total_area = area_i.sum(dim=1)  # [B, M]
        total_load = load_i.sum(dim=1)  # [B, M]

        return total_load, total_area