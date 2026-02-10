import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import load_config
from physics.differentiable import AxisymmetricContactLayer
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
        
        # since we sort heights later, Index 0 will always be the active one.
        return n, h

    def generate_canonical_walls(self, n_samples):
        """
        The "Stiffest Limits".
        generates a mix of Perfect Walls (h=0) and Quasi-Walls (h ~ epsilon).
        This helps the NN understand the transition to the limit.
        """
        print(f"  > Generating {n_samples} Canonical & Quasi-Walls...")
        n, h = self.get_base_batch(n_samples)
        
        # 1. Sweep Exponents (1.0 to 8.0)
        n_vals = torch.linspace(1.0, 8.0, n_samples).unsqueeze(1)
        n = n_vals.repeat(1, self.n_asp)
        
        # 2. Perfect Walls (First 50%)
        # These are the theoretical maximums
        split_idx = n_samples // 2
        h[:split_idx].fill_(0.0)
        

        roughness_scale = 0.02 * self.max_delta
        
        # Generate random tiny heights
        micro_noise = torch.rand(n_samples - split_idx, self.n_asp) * roughness_scale
        h[split_idx:] = micro_noise
        
        # 4. Sort and Normalize (Critical)
        # Even Quasi-walls must follow the sorted convention
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]
        
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
            n_active = self.n_asp // 4
            
            h_active = torch.rand(n_active) * (0.05 *self.max_delta) # Very tight contact
            h_inactive = self.max_delta * 1.1 * torch.ones(self.n_asp - n_active)        # (0.5 + 0.5 * torch.rand(self.n_asp - n_active))
            
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
        n_singles = int(0.10 * total_samples)  # 10% Pure Singles (Basis)
        n_walls = int(0.05 * total_samples)    # 5% Pure Walls 
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
    

if __name__ == "__main__":
    # 1. Setup
    print("--- Starting Dataset Generation ---")
    
    # Load Config
    # We assume this script is located in 'data/' and config is in root
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    cfg = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Generate Parameters (The "Y" data)
    gen = SurfaceGenerator(cfg)
    n_samples = cfg['data']['n_samples']
    
    print(f"Generating {n_samples} mixed surface parameters...")
    all_n, all_h = gen.mix_dataset(total_samples=n_samples)
    
    # Move to device for physics calculation
    all_n = all_n.to(device)
    all_h = all_h.to(device)
    
    # 3. Solve Physics (The "X" data)
    print("Solving physics to generate Load/Area curves...")
    phys = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    
    # Constants
    R = cfg['physics']['radius']
    max_d = cfg['physics']['max_delta_ratio'] * R
    n_steps = cfg['data']['n_steps']
    
    indentations = torch.linspace(0, max_d, n_steps).to(device).unsqueeze(0)
    t_w = torch.ones(1, cfg['physics']['n_asperities']).to(device) * 2.0 * R
    
    # Batch Processing
    batch_size = 1000 
    dataset = TensorDataset(all_n, all_h)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_loads = []
    all_areas = []
    all_stiff = []
    
    for batch_n, batch_h in tqdm(loader, desc="Physics Engine"):
        with torch.no_grad():
            current_batch = batch_n.shape[0]
            batch_ind = indentations.repeat(current_batch, 1)
            
            # Solve
            load, area = phys(batch_h, batch_n, t_w, batch_ind)
            
            # Stiffness
            stiffness = torch.diff(load, dim=1, prepend=torch.zeros(current_batch, 1).to(device))
            
            all_loads.append(load.cpu())
            all_areas.append(area.cpu())
            all_stiff.append(stiffness.cpu())
            
    # Concatenate results
    X_final = torch.stack([
        torch.cat(all_loads, dim=0),
        torch.cat(all_areas, dim=0),
        torch.cat(all_stiff, dim=0)
    ], dim=1)
    
    Y_final = torch.cat([all_n.cpu(), all_h.cpu()], dim=1)
    
    # 4. Save using Config Path
    # We resolve the path relative to the project root (Current Working Directory)
    save_path = cfg['data']['path']
    
    # Ensure the directory exists (e.g., if path is 'data/subdir/dataset.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Saving dataset to: {save_path}")
    torch.save({
        "x": X_final,
        "y": Y_final 
    }, save_path)
    
    print("--- Dataset Generation Complete ---")