import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats.qmc import LatinHypercube

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config
from utils.seeding import set_seed
from physics.differentiable import AxisymmetricContactLayer

class SurfaceGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_asp = cfg['physics']['n_asperities']
        self.max_delta = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']

    def get_base_batch(self, n_samples):
        return torch.zeros(n_samples, self.n_asp), torch.zeros(n_samples, self.n_asp)

    def generate_exiled_contacts(self, n_samples):
        print(f"  > Generating {n_samples} Exiled Contacts (Learning to hide asperities)...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 2.0 

        for i in range(n_samples):
            # Only 1 to 3 asperities actually make contact
            n_active = np.random.randint(1, 4)
            h_active = torch.rand(n_active) * (0.5 * self.max_delta)
            # The rest are exiled beyond max_delta so they NEVER touch
            h_exiled = (1.1 + torch.rand(self.n_asp - n_active)) * self.max_delta
            combined = torch.cat([h_active, h_exiled])
            h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        n = torch.gather(n, 1, sorted_idx)
        h = h - h[:, 0:1]
        return n, h

    def generate_bimodal_extremes(self, n_samples):
        print(f"  > Generating {n_samples} Bimodal Extremes (Violent stiffness cliffs)...")
        n, h = self.get_base_batch(n_samples)
        
        for i in range(n_samples):
            # Half soft cones early, half blunt cubes late
            split = self.n_asp // 2
            
            n_soft = 1.0 + torch.rand(split) * 0.5
            h_soft = torch.rand(split) * (0.2 * self.max_delta)
            
            n_blunt = 2.5 + torch.rand(self.n_asp - split) * 0.5
            # Placed deep so they hit like a wall later
            h_blunt = (0.5 + torch.rand(self.n_asp - split) * 0.4) * self.max_delta
            
            combined_h = torch.cat([h_soft, h_blunt])
            combined_n = torch.cat([n_soft, n_blunt])
            
            perm = torch.randperm(self.n_asp)
            h[i] = combined_h[perm]
            n[i] = combined_n[perm]

        h, sorted_idx = torch.sort(h, dim=1)
        n = torch.gather(n, 1, sorted_idx)
        h = h - h[:, 0:1]
        return n, h

    def generate_canonical_walls(self, n_samples):
        print(f"  > Generating {n_samples} Canonical Walls ...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 2.0 

        for i in range(n_samples):
            mode = np.random.rand()
            if mode < 0.5:
                # All 9 touch almost immediately
                roughness = torch.rand(self.n_asp) * (0.01 * self.max_delta)
                h[i] = roughness
            else:
                n_active = np.random.randint(4, self.n_asp) 
                h_active = torch.rand(n_active) * (0.01 * self.max_delta)
                h_inactive = (0.5 + 0.5 * torch.rand(self.n_asp - n_active)) * self.max_delta
                combined = torch.cat([h_active, h_inactive])
                h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        n = torch.gather(n, 1, sorted_idx)
        h = h - h[:, 0:1]
        return n, h

    def generate_sparse(self, n_samples):
        print(f"  > Generating {n_samples} Sparse (Sequential) samples...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 2.0

        for i in range(n_samples):
            n_active = np.random.randint(2, max(3, self.n_asp // 2))
            h_active = torch.rand(n_active) * (0.8 * self.max_delta)
            h_inactive = (0.8 + 0.2 * torch.rand(self.n_asp - n_active)) * self.max_delta
            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        n = torch.gather(n, 1, sorted_idx)
        h = h - h[:, 0:1]
        return n, h

    def generate_random_sums(self, n_samples):
        print(f"  > Generating {n_samples} Random Sums ...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 2.0

        for i in range(n_samples):
            n_active = np.random.randint(1, self.n_asp + 1)
            mode = np.random.rand()

            if mode < 0.33:
                h_active = torch.rand(n_active) * self.max_delta
            elif mode < 0.66:
                mean_depth = np.random.rand() * self.max_delta
                sigma = 0.1 * self.max_delta
                h_active = torch.normal(mean_depth, sigma, size=(n_active,))
            else:
                raw_exp = torch.distributions.Exponential(rate=3.0).sample((n_active,))
                h_active = raw_exp * (0.3 * self.max_delta)

            h_active = torch.clamp(h_active, 0, self.max_delta)
            h_inactive = (1.0 + 0.2 * torch.rand(self.n_asp - n_active)) * self.max_delta

            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        n = torch.gather(n, 1, sorted_idx)
        h = h - h[:, 0:1]
        return n, h

    def mix_dataset(self, total_samples):
        # We reserve 10% for Exiled, 10% for Bimodal, 10% for Walls, 10% for Sparse
        n_exiled = int(0.10 * total_samples)
        n_bimodal = int(0.10 * total_samples)
        n_wall = int(0.10 * total_samples)
        n_sparse = int(0.10 * total_samples)
        n_rnd = int(0.30 * total_samples) 
        n_lhs = total_samples - (n_exiled + n_bimodal + n_wall + n_sparse + n_rnd)

        print("--- Mixing Dataset Sub-Domains ---")
        sampler = LatinHypercube(d=2*self.n_asp)
        sample = sampler.random(n=n_lhs)
        n_lhs_data = 1.0 + torch.tensor(sample[:, :self.n_asp]).float() * 2.0
        h_lhs_data = torch.tensor(sample[:, self.n_asp:]).float() * self.max_delta

        n_rn, h_rn = self.generate_random_sums(n_rnd)
        n_ex, h_ex = self.generate_exiled_contacts(n_exiled)
        n_bi, h_bi = self.generate_bimodal_extremes(n_bimodal)
        n_wa, h_wa = self.generate_canonical_walls(n_wall)
        n_sp, h_sp = self.generate_sparse(n_sparse)

        all_n = torch.cat([n_lhs_data, n_rn, n_ex, n_bi, n_wa, n_sp])
        all_h = torch.cat([h_lhs_data, h_rn, h_ex, h_bi, h_wa, h_sp])

        all_h, sorted_idx = torch.sort(all_h, dim=1)
        all_n = torch.gather(all_n, 1, sorted_idx)
        all_h = all_h - all_h[:, 0:1]

        return all_n, all_h


if __name__ == "__main__":
    print("--- Starting Dataset Generation ---")
    set_seed(42)

    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gen = SurfaceGenerator(cfg)
    target_samples = cfg['data']['n_samples']

    all_n, all_h = gen.mix_dataset(total_samples=target_samples)
    all_n = all_n.to(device)
    all_h = all_h.to(device)

    phys = AxisymmetricContactLayer(cfg=cfg).to(device) 

    R = cfg['physics']['radius']
    max_d = cfg['physics']['max_delta_ratio'] * R
    n_steps = cfg['data']['n_steps']
    n_asp = cfg['physics']['n_asperities']

    indentations = torch.linspace(0, max_d, n_steps).to(device).unsqueeze(0)
    t_w = torch.ones(1, n_asp).to(device) * 2.0 * R

    # ---------------------------------------------------------
    # 1. ESTABLISH THE GLOBAL PHYSICAL BOUNDING BOX (P*_max)
    # ---------------------------------------------------------
    print("Calculating Global Maximum Nominal Capacity (P*_max)...")
    # The absolute ceiling: All asperities are n=3 (bluntest), all at h=0 (engaging instantly)
    n_ceiling = torch.ones(1, n_asp).to(device) * 3.0
    h_ceiling = torch.zeros(1, n_asp).to(device)
    
    with torch.no_grad():
        P_bound, _, _ = phys(h_ceiling, n_ceiling, t_w, indentations, k_steepness=1e6)
        global_P_max = P_bound[0, -1].item()
    
    print(f"Global P*_max established at: {global_P_max:.6f}")
    
    # Define the standardized fixed grid
    p_star_grid = torch.linspace(0, global_P_max, n_steps)
    p_star_grid_np = p_star_grid.numpy()

    # ---------------------------------------------------------
    # 2. RUN PHYSICS AND INTERPOLATE ONTO FIXED GRID
    # ---------------------------------------------------------
    print("Solving physics and interpolating to P* grid...")
    batch_size = 2000
    dataset = TensorDataset(all_n, all_h)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_P_grid = []
    all_alpha_padded = []
    all_stiff_padded = []

    with torch.no_grad():
        for batch_n, batch_h in loader:
            batch_n = batch_n.to(device)
            batch_h = batch_h.to(device)
            current_batch = batch_n.shape[0]
            batch_ind = indentations.repeat(current_batch, 1)
            batch_tw = t_w.repeat(current_batch, 1)

            # Generate native curves in displacement space
            P_native, alpha_native, stiff_native = phys(batch_h, batch_n, batch_tw, batch_ind, k_steepness=1e6)
            
            P_np = P_native.cpu().numpy()
            alpha_np = alpha_native.cpu().numpy()
            stiff_np = stiff_native.cpu().numpy()

            # Interpolate each sample onto the fixed P* grid
            batch_alpha_interp = np.zeros((current_batch, n_steps))
            batch_stiff_interp = np.zeros((current_batch, n_steps))
            
            for i in range(current_batch):
                # numpy.interp with right=-1.0 perfectly creates our padded mask out-of-bounds
                batch_alpha_interp[i] = np.interp(p_star_grid_np, P_np[i], alpha_np[i], right=-1.0)
                batch_stiff_interp[i] = np.interp(p_star_grid_np, P_np[i], stiff_np[i], right=-1.0)

            # Since the P grid is fixed, we just repeat the standard grid for all samples
            batch_P_grid = p_star_grid.unsqueeze(0).repeat(current_batch, 1)

            all_P_grid.append(batch_P_grid)
            all_alpha_padded.append(torch.tensor(batch_alpha_interp, dtype=torch.float32))
            all_stiff_padded.append(torch.tensor(batch_stiff_interp, dtype=torch.float32))

    # Assemble Final Tensor
    # Channel 0: P* (Fixed Grid)
    # Channel 1: Alpha (Padded with -1.0)
    # Channel 2: Stiffness (Padded with -1.0)
    X_final = torch.stack([
        torch.cat(all_P_grid, dim=0),
        torch.cat(all_alpha_padded, dim=0),
        torch.cat(all_stiff_padded, dim=0) 
    ], dim=1)

    Y_final = torch.cat([all_n.cpu(), all_h.cpu()], dim=1)

    save_path = cfg['data']['path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "x": X_final,
        "y": Y_final,
        "p_star_max": global_P_max  # Save this so the training script knows the normalization bound
    }, save_path)

    print("--- Dataset Generation Complete ---")