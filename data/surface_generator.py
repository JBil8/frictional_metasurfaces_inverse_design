import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube

class SurfaceGenerator:
    def __init__(self, config):
        self.cfg = config
        self.n_asp = config['physics']['n_asperities']
        self.max_delta = config['physics']['max_delta_ratio'] * config['physics']['radius']
        
    def get_base_batch(self, n_samples):
        return torch.zeros(n_samples, self.n_asp), torch.zeros(n_samples, self.n_asp)

    def generate_canonical_singles(self, n_samples):
        """
        The "Basis Functions". 
        Generates surfaces with EXACTLY ONE active asperity.
        We sweep through exponents n=[1, 1.5, ... 8].
        """
        print(f"  > Generating {n_samples} Canonical Singles (Basis Functions)...")
        n, h = self.get_base_batch(n_samples)
        
        # Set all heights to MAX (infinity)
        h.fill_(self.max_delta)
        
        # Set the FIRST asperity to 0.0 (Active)
        h[:, 0] = 0.0
        
        # Sweep exponents from 1.0 to 8.0 across the batch
        # This teaches the network: "What does a single n=1 look like? What does a single n=8 look like?"
        n_vals = torch.linspace(1.0, 8.0, n_samples).unsqueeze(1)
        n = n_vals.repeat(1, self.n_asp)
        
        # (Optional) Randomize the index of the active asperity so it doesn't memorize "Index 0"
        # But since we sort heights later, Index 0 will always be the active one.
        return n, h

    def generate_canonical_walls(self, n_samples):
        """
        The "Stiffest Limits".
        Generates surfaces where ALL asperities touch at once (h=0).
        """
        print(f"  > Generating {n_samples} Canonical Walls...")
        n, h = self.get_base_batch(n_samples)
        
        # All heights = 0 (Wall)
        h.fill_(0.0)
        
        # Sweep exponents
        n_vals = torch.linspace(1.0, 8.0, n_samples).unsqueeze(1)
        n = n_vals.repeat(1, self.n_asp)
        
        return n, h

    def generate_sparse(self, n_samples):
        """
        Mostly air. 1 to 3 asperities active.
        """
        print(f"  > Generating {n_samples} Sparse samples...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 7.0
        
        for i in range(n_samples):
            # Reduced count: Strictly 1 to 3 active
            n_active = np.random.randint(1, 4)
            
            h_active = torch.rand(n_active) * (0.05 * self.max_delta) # Very tight contact
            h_inactive = self.max_delta * (0.5 + 0.5 * torch.rand(self.n_asp - n_active))
            
            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]
        return n, h

    def generate_bimodal(self, n_samples):
        """
        Friction Switches.
        Tweaked to allow '1 vs 15' splits (Extreme Slip).
        """
        print(f"  > Generating {n_samples} Bimodal (Switch) samples...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 7.0
        
        for i in range(n_samples):
            gap = np.random.uniform(0.2, 0.8) * self.max_delta
            
            # CHANGED: Allow split_idx to be 1 (Extreme case: 1 asperity carries load, then 15 hit)
            split_idx = np.random.randint(1, self.n_asp - 1)
            
            g1 = torch.normal(0, 0.01 * self.max_delta, size=(split_idx,))
            g2 = torch.normal(gap, 0.01 * self.max_delta, size=(self.n_asp - split_idx,))
            
            combined = torch.cat([g1, g2])
            h[i] = torch.clamp(combined, 0, self.max_delta)
            
        return n, h
    
    def mix_dataset(self, total_samples=100000):
        # Adjusted Ratios to emphasize "Physics Basis"
        n_lhs = int(0.50 * total_samples)      # 50% General Noise
        n_singles = int(0.10 * total_samples)  # 10% Pure Singles (Basis) [NEW]
        n_walls = int(0.05 * total_samples)    # 5% Pure Walls [NEW]
        n_sparse = int(0.15 * total_samples)   # 15% Sparse
        n_bimodal = int(0.20 * total_samples)  # 20% Switches
        
        # 1. Standard LHS
        print(f"Generating {n_lhs} LHS samples...")
        sampler = LatinHypercube(d=2*self.n_asp)
        sample = sampler.random(n=n_lhs)
        n_lhs_data = 1.0 + torch.tensor(sample[:, :self.n_asp]).float() * 7.0
        h_lhs_data = torch.tensor(sample[:, self.n_asp:]).float() * self.max_delta
        
        # 2. Exotic
        n_si, h_si = self.generate_canonical_singles(n_singles)
        n_wa, h_wa = self.generate_canonical_walls(n_walls)
        n_sp, h_sp = self.generate_sparse(n_sparse)
        n_bi, h_bi = self.generate_bimodal(n_bimodal)
        
        # 3. Concatenate
        all_n = torch.cat([n_lhs_data, n_si, n_wa, n_sp, n_bi])
        all_h = torch.cat([h_lhs_data, h_si, h_wa, h_sp, h_bi])
        
        # 4. Sort & Normalize
        all_h, _ = torch.sort(all_h, dim=1)
        all_h = all_h - all_h[:, 0:1] # Normalize relative to first contact
        
        return all_n, all_h