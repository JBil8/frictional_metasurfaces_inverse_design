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
        
        # Set all heights to MAX so they do not touch
        h.fill_(self.max_delta)
        
        # Set the first asperity to 0.0 (Active)
        h[:, 0] = 0.0
        
        # Sweep exponents from 1.0 to 8.0 across the batch
        n_vals = torch.linspace(1.0, 8.0, n_samples).unsqueeze(1)
        n = n_vals.repeat(1, self.n_asp)
        
        return n, h

    def generate_canonical_walls(self, n_samples):
        """
        Generates Stiff Limits (n=8).
        Includes:
        1. Solid Block (All h=0)
        2. Quasi-Block (h ~ epsilon)
        3. Partial Walls (Some h=0, some h=infinity) -> Teaches 'Local' Wall behavior
        """
        print(f"  > Generating {n_samples} Canonical Walls ...")
        n, h = self.get_base_batch(n_samples)
        
        # We vary slightly [7.0, 8.0] to make it robust
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 1.0
        
        # 2. Logic Split
        for i in range(n_samples):
            mode = np.random.rand()
            
            if mode < 0.5:
                # All heights are roughly 0
                roughness = torch.rand(self.n_asp) * (0.01 * self.max_delta)
                h[i] = roughness
                
            else:
                # This teaches: "n=8 matters even if only 3 asperities touch."
                n_active = np.random.randint(1, self.n_asp) # 1 to 15 active
                
                # Active: At 0
                h_active = torch.rand(n_active) * (0.01 * self.max_delta)
                # Inactive: Far away
                h_inactive = (0.5 + 0.5 * torch.rand(self.n_asp - n_active)) * self.max_delta
                
                combined = torch.cat([h_active, h_inactive])
                h[i] = combined[torch.randperm(self.n_asp)]
        
        # Sort & Normalize
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]
        
        return n, h

    def generate_sparse(self, n_samples):
        """
        Sparse / Sequential Contacts.
        Focuses on distinct, isolated contact events to teach the NN 
        how to resolve "Kinks" in the load curve.
        """
        print(f"  > Generating {n_samples} Sparse (Sequential) samples...")
        n, h = self.get_base_batch(n_samples)
        
        # Exponents can be anything
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 7.0
        
        for i in range(n_samples):
            # STRICTLY LOW COUNT: 2 to 4 active.
            # If we have too many, it becomes a "Random Sum" surface.
            n_active = np.random.randint(2, 5) 
            
            # SPREAD OUT: Place them randomly across the gap
            # This ensures they contact one-by-one (Sequential)
            h_active = torch.rand(n_active) * (0.8 * self.max_delta)
            
            # Inactive: Pushed away
            h_inactive = (1.0 + 0.2 * torch.rand(self.n_asp - n_active)) * self.max_delta
            
            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]
        
        # Sort & Normalize
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]

        return n, h

    def generate_multistep(self, n_samples):
        """
        Generates 'Staircase' surfaces with multiple discrete height levels.
        (Generalization of the Bimodal Switch).
        
        Logic:
        1. Determine how many steps (levels) to have.
        2. Assign asperities to these levels.
        3. Add small noise so they aren't mathematically perfect (robustness).
        """
        print(f"  > Generating {n_samples} Multi-Step (Staircase) samples...")
        n, h = self.get_base_batch(n_samples)
        
        # Exponents: Mix of Walls (flat) and Spheres (curved) to make steps interesting
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 7.0
        
        for i in range(n_samples):
            # 1. How many levels? (User's heuristic: n // 4)
            # Ensure at least 2 levels (otherwise it's just a flat wall)
            max_levels = max(2, self.n_asp // 4)
            
            # Randomly pick between 2 and max_levels
            n_levels = np.random.randint(2, max_levels + 1)
            
            # 2. Pick the Heights for these levels
            # We pick random points in [0, max_delta] and sort them.
            # The first level is forced to 0 (physics normalization).
            level_heights = torch.rand(n_levels) * (0.9 * self.max_delta)
            level_heights[:1] = 0.0 
            level_heights, _ = torch.sort(level_heights)
            
            # 3. Distribute Asperities across these levels
            # We need to split 'n_asp' items into 'n_levels' groups.
            # We do this by picking (n_levels - 1) random "cut points".
            if n_levels < self.n_asp:
                # Pick unique cut points
                cuts = np.sort(np.random.choice(range(1, self.n_asp), n_levels - 1, replace=False))
                bounds = np.concatenate(([0], cuts, [self.n_asp]))
            else:
                # Fallback if n_asp is tiny
                bounds = np.arange(n_levels + 1)

            # 4. Construct the Surface
            combined_h = []
            for j in range(n_levels):
                count = bounds[j+1] - bounds[j]
                if count > 0:
                    # Base height + Micro-roughness (Jitter)
                    # We add jitter so the network doesn't overfit to "perfect" steps
                    jitter = torch.randn(count) * (0.01 * self.max_delta)
                    group_h = level_heights[j] + jitter
                    combined_h.append(group_h)
            
            # Concatenate and clamp
            if combined_h:
                h_seq = torch.cat(combined_h)
                # Pad if calculation was slightly off (safety)
                if len(h_seq) < self.n_asp:
                    padding = torch.ones(self.n_asp - len(h_seq)) * self.max_delta
                    h_seq = torch.cat([h_seq, padding])
                
                h[i] = torch.clamp(h_seq, 0, self.max_delta)

        # Final Sort & Normalize (Crucial for invariance)
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]
        
        return n, h
 
    def generate_random_sums(self, n_samples):
        """
        Teaches the NN the 'Principle of Superposition' and 'Continuous Heights'.
        1. Vary n_active from 1 to N.
        2. Place active asperities continuously across [0, max_delta], not just at 0.
        """
        print(f"  > Generating {n_samples} Random Sums ...")
        n, h = self.get_base_batch(n_samples)
        
        # Random Exponents (1.0 to 8.0)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 7.0
        
        for i in range(n_samples):
            # complexity (4-12) which is rare in LHS
            n_active = np.random.randint(1, self.n_asp + 1)
            
            # 2. Pick Heights: The Generalization Key
            mode = np.random.rand()
            
            if mode < 0.33:
                # Mode A: Uniform Random (The "LHS" look, but simpler)
                # Active are anywhere in [0, max_delta]
                active_h = torch.rand(n_active) * self.max_delta
                
            elif mode < 0.66:
                # Mode B: Gaussian Cluster (The "Wall/Switch" look)
                # Clustered around a random mean depth
                mean_depth = np.random.rand() * self.max_delta
                sigma = 0.1 * self.max_delta
                active_h = torch.normal(mean_depth, sigma, size=(n_active,))
                
            else:
                # Mode C: Exponential-ish (The "Linear/Coulomb" look) [NEW]
                # This explicitly creates the "Long Tail" the network was missing.
                # Many at 0, fewer at 0.2, very few at 0.5...
                raw_exp = torch.distributions.Exponential(rate=3.0).sample((n_active,))
                # Scale to physical range
                active_h = raw_exp * (0.3 * self.max_delta)
            
            # 3. Clip to stay physical
            active_h = torch.clamp(active_h, 0, self.max_delta)
            
            # 4. Inactive ones pushed to infinity
            # (But vary "infinity" slightly so the network doesn't memorize a specific value)
            inactive_h = (1.2 + 0.5 * torch.rand(self.n_asp - n_active)) * self.max_delta
            
            combined = torch.cat([active_h, inactive_h])
            h[i] = combined[torch.randperm(self.n_asp)]
            
        # Sort & Normalizew
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]
        
        return n, h

    def mix_dataset(self, total_samples=None):
        if total_samples is None: total_samples = self.cfg['data']['n_samples']
        ratios = self.cfg['generation']['ratios']
        
        # Calculate counts
        n_lhs = int(ratios['lhs'] * total_samples)
        n_rnd = int(ratios['random_sum'] * total_samples) # NEW
        n_single = int(ratios['single'] * total_samples)
        n_wall = int(ratios['wall'] * total_samples)
        n_sparse = int(ratios['sparse'] * total_samples)
        n_switch = int(ratios['switch'] * total_samples)
        
        # Fix rounding (Assign remainder to LHS)
        current_sum = n_lhs + n_rnd + n_single + n_wall + n_sparse + n_switch
        n_lhs += (total_samples - current_sum)
        
        print(f"Generating Dataset ({total_samples} samples):")
        print(f"  - LHS:        {n_lhs}")
        print(f"  - Random Sum: {n_rnd}")
        print(f"  - Single:     {n_single}")
        print(f"  - Wall:       {n_wall}")
        print(f"  - Sparse:     {n_sparse}")
        print(f"  - Switch:     {n_switch}")
        
        # 1. LHS
        sampler = LatinHypercube(d=2*self.n_asp)
        sample = sampler.random(n=n_lhs)
        n_lhs_data = 1.0 + torch.tensor(sample[:, :self.n_asp]).float() * 7.0
        h_lhs_data = torch.tensor(sample[:, self.n_asp:]).float() * self.max_delta
        
        # 2. Others
        n_rn, h_rn = self.generate_random_sums(n_rnd) # NEW
        n_si, h_si = self.generate_canonical_singles(n_single)
        n_wa, h_wa = self.generate_canonical_walls(n_wall)
        n_sp, h_sp = self.generate_sparse(n_sparse)
        n_bi, h_bi = self.generate_multistep(n_switch)
        
        # 3. Concatenate (Order matters for Index Ranges!)
        # Order: LHS -> Random Sum -> Single -> Wall -> Sparse -> Switch
        all_n = torch.cat([n_lhs_data, n_rn, n_si, n_wa, n_sp, n_bi])
        all_h = torch.cat([h_lhs_data, h_rn, h_si, h_wa, h_sp, h_bi])
        
        # 4. Final Sort & Normalize
        all_h, _ = torch.sort(all_h, dim=1)
        all_h = all_h - all_h[:, 0:1] 
        
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