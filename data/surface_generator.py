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
    

if __name__ == "__main__":
    # 1. Setup
    print("--- Starting Dataset Generation ---")
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
    
    phys = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    
    # Prepare constants
    R = cfg['physics']['radius']
    n_asp = cfg['physics']['n_asperities']
    max_d = cfg['physics']['max_delta_ratio'] * R
    n_steps = cfg['data']['n_steps']
    
    # Indentation history (same for all samples)
    indentations = torch.linspace(0, max_d, n_steps).to(device).unsqueeze(0) # [1, steps]
    t_w = torch.ones(1, n_asp).to(device) * 2.0 * R
    
    # Run in batches to avoid OOM
    batch_size = 1000 # Physics batch size
    dataset = TensorDataset(all_n, all_h)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_loads = []
    all_areas = []
    all_stiff = []
    
    for batch_n, batch_h in tqdm(loader, desc="Physics Engine"):
        with torch.no_grad():
            # Expand indents to batch size
            current_batch = batch_n.shape[0]
            batch_ind = indentations.repeat(current_batch, 1)
            
            # Solve
            load, area = phys(batch_h, batch_n, t_w, batch_ind)
            
            # Calculate Stiffness (dL/d_delta)
            # Simple difference method
            stiffness = torch.diff(load, dim=1, prepend=torch.zeros(current_batch, 1).to(device))
            
            all_loads.append(load.cpu())
            all_areas.append(area.cpu())
            all_stiff.append(stiffness.cpu())
            
    # Concatenate results
    X_load = torch.cat(all_loads, dim=0) # [N, Steps]
    X_area = torch.cat(all_areas, dim=0)
    X_stiff = torch.cat(all_stiff, dim=0)
    
    # Stack into [N, 3, Steps] format for CNN input
    X_final = torch.stack([X_load, X_area, X_stiff], dim=1)
    
    # Concatenate parameters for Y: [N, 32] (16 exponents + 16 heights)
    Y_final = torch.cat([all_n.cpu(), all_h.cpu()], dim=1)
    
    # 4. Save
    # We save to a NEW filename to distinguish from the old LHS dataset
    save_path = os.path.join(os.path.dirname(__file__), "dataset_16_asp_mixed.pt")
    
    print(f"Saving dataset to {save_path}...")
    torch.save({
        "x": X_final, # Inputs: (Load, Area, Stiffness)
        "y": Y_final  # Targets: (n, h)
    }, save_path)
    
    print("--- Dataset Generation Complete ---")