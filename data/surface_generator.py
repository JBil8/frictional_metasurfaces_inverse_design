import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube

class SurfaceGenerator:
    def __init__(self, config):
        self.cfg = config
        self.n_asp = config['physics']['n_asperities']
        self.max_delta = config['physics']['max_delta_ratio'] * config['physics']['radius']
        
    def get_base_batch(self, n_samples):
        """Helper to get empty containers"""
        # Exponents n in [1, 8]
        # Heights h in [0, max_delta]
        return torch.zeros(n_samples, self.n_asp), torch.zeros(n_samples, self.n_asp)

    def generate_sparse(self, n_samples):
        """
        Generates surfaces where mostly 'air' exists.
        Only k asperities touch. The rest are buried deep.
        """
        print(f"  > Generating {n_samples} Sparse samples...")
        n, h = self.get_base_batch(n_samples)
        
        # Random exponents [1, 8]
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 7.0
        
        for i in range(n_samples):
            # Pick how many active: 1 to 4
            n_active = np.random.randint(1, 5)
            
            # Active ones: h ~ 0 (Touch immediately)
            h_active = torch.rand(n_active) * (0.1 * self.max_delta)
            
            # Inactive ones: h ~ max_delta (Don't touch until end)
            h_inactive = self.max_delta * (0.8 + 0.2 * torch.rand(self.n_asp - n_active))
            
            # Combine and Shuffle
            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]
            
        return n, h

    def generate_bimodal(self, n_samples):
        """
        Generates 'Friction Switches'.
        Group A at height 0. Group B at height 'gap'.
        """
        print(f"  > Generating {n_samples} Bimodal (Switch) samples...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 7.0
        
        for i in range(n_samples):
            # Define the gap size
            gap = np.random.uniform(0.2, 0.7) * self.max_delta
            
            # Split: 30% active vs 70% delayed (or random split)
            split_idx = np.random.randint(2, self.n_asp - 2)
            
            # Group 1: h ~ 0
            g1 = torch.normal(0, 0.01 * self.max_delta, size=(split_idx,))
            
            # Group 2: h ~ gap
            g2 = torch.normal(gap, 0.01 * self.max_delta, size=(self.n_asp - split_idx,))
            
            combined = torch.cat([g1, g2])
            # Clip to valid range
            h[i] = torch.clamp(combined, 0, self.max_delta)
            
        return n, h
    
    def generate_uniform_exponents(self, n_samples, value=None):
        """
        Forces all asperities to have the SAME shape (e.g., all cones).
        Helps learn the 'pure' physics boundaries.
        """
        print(f"  > Generating {n_samples} Homogeneous samples...")
        _, h = self.get_base_batch(n_samples)
        
        # Random heights (LHS style)
        sampler = LatinHypercube(d=self.n_asp)
        h = torch.tensor(sampler.random(n_samples), dtype=torch.float32) * self.max_delta
        
        if value is None:
            # Random fixed value per sample (e.g., row 1 is all 2.5, row 2 is all 7.1)
            vals = 1.0 + torch.rand(n_samples, 1) * 7.0
            n = vals.repeat(1, self.n_asp)
        else:
            n = torch.ones(n_samples, self.n_asp) * value
            
        return n, h

    def mix_dataset(self, total_samples=100000):
        """
        Creates the final mixed dataset.
        """
        n_lhs = int(0.6 * total_samples)
        n_sparse = int(0.15 * total_samples)
        n_bimodal = int(0.15 * total_samples)
        n_pure = total_samples - n_lhs - n_sparse - n_bimodal
        
        # 1. Standard LHS
        print(f"Generating {n_lhs} LHS samples...")
        # (Insert your existing LHS logic here)
        sampler = LatinHypercube(d=2*self.n_asp)
        sample = sampler.random(n=n_lhs)
        n_lhs_data = 1.0 + torch.tensor(sample[:, :self.n_asp]).float() * 7.0
        h_lhs_data = torch.tensor(sample[:, self.n_asp:]).float() * self.max_delta
        
        # 2. Exotic 
        n_sp, h_sp = self.generate_sparse(n_sparse)
        n_bi, h_bi = self.generate_bimodal(n_bimodal)
        n_pu, h_pu = self.generate_uniform_exponents(n_pure) # Learns pure cones/punches
        
        # 3. Concatenate
        all_n = torch.cat([n_lhs_data, n_sp, n_bi, n_pu])
        all_h = torch.cat([h_lhs_data, h_sp, h_bi, h_pu])
        
        # 4. Sort Heights (CRITICAL for Canonical Ordering)
        # We sort h, and we effectively don't care which n moves where 
        # because n is random in most cases. 
        # (If n was correlated to h, we'd need argsort. Here independent is fine for training)
        all_h, _ = torch.sort(all_h, dim=1)
        
        # Normalize h relative to first contact
        all_h = all_h - all_h[:, 0:1]
        
        return all_n, all_h