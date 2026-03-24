from utils.seeding import set_seed
from physics.differentiable import AxisymmetricContactLayer
from utils.config import load_config
import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from scipy.stats.qmc import LatinHypercube

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SurfaceGenerator:
    def __init__(self, config):
        self.cfg = config
        self.n_asp = config['physics']['n_asperities']
        self.max_delta = config['physics']['max_delta_ratio'] * \
            config['physics']['radius']

    def get_base_batch(self, n_samples):
        return torch.zeros(n_samples, self.n_asp), torch.zeros(n_samples, self.n_asp)

    def generate_canonical_singles(self, n_samples):
        print(f"  > Generating {n_samples} Canonical Singles (Basis Functions)...")
        n, h = self.get_base_batch(n_samples)
        h.fill_(self.max_delta)
        h[:, 0] = 0.0
        n_vals = torch.linspace(1.0, 3.0, n_samples).unsqueeze(1)
        n = n_vals.repeat(1, self.n_asp)
        return n, h

    def generate_canonical_walls(self, n_samples):
        print(f"  > Generating {n_samples} Canonical Walls ...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 1.0

        for i in range(n_samples):
            mode = np.random.rand()
            if mode < 0.5:
                roughness = torch.rand(self.n_asp) * (0.01 * self.max_delta)
                h[i] = roughness
            else:
                n_active = np.random.randint(1, self.n_asp) 
                h_active = torch.rand(n_active) * (0.01 * self.max_delta)
                h_inactive = (1.0 + 0.2 * torch.rand(self.n_asp - n_active)) * self.max_delta
                combined = torch.cat([h_active, h_inactive])
                h[i] = combined[torch.randperm(self.n_asp)]

        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]
        return n, h

    def generate_sparse(self, n_samples):
        print(f"  > Generating {n_samples} Sparse (Sequential) samples...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 2.0

        for i in range(n_samples):
            n_active = np.random.randint(2, max(3, self.n_asp // 2))
            h_active = torch.rand(n_active) * (0.8 * self.max_delta)
            h_inactive = (1.0 + 0.2 * torch.rand(self.n_asp - n_active)) * self.max_delta
            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]

        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]
        return n, h

    def generate_multistep(self, n_samples):
        print(f"  > Generating {n_samples} Multi-Step (Staircase) samples...")
        n, h = self.get_base_batch(n_samples)
        n = 1.0 + torch.rand(n_samples, self.n_asp) * 2.0

        for i in range(n_samples):
            max_levels = max(2, self.n_asp // 3)
            n_levels = np.random.randint(2, max_levels + 1)

            level_heights = torch.rand(n_levels) * (0.9 * self.max_delta)
            level_heights[:1] = 0.0
            level_heights, _ = torch.sort(level_heights)

            if n_levels < self.n_asp:
                cuts = np.sort(np.random.choice(range(1, self.n_asp), n_levels - 1, replace=False))
                bounds = np.concatenate(([0], cuts, [self.n_asp]))
            else:
                bounds = np.arange(n_levels + 1)

            combined_h = []
            for j in range(n_levels):
                count = bounds[j+1] - bounds[j]
                if count > 0:
                    jitter = torch.randn(count) * (0.01 * self.max_delta)
                    group_h = level_heights[j] + jitter
                    combined_h.append(group_h)

            if combined_h:
                h_seq = torch.cat(combined_h)
                if len(h_seq) < self.n_asp:
                    padding = torch.ones(self.n_asp - len(h_seq)) * self.max_delta
                    h_seq = torch.cat([h_seq, padding])
                h[i] = torch.clamp(h_seq, 0, self.max_delta)

        h, _ = torch.sort(h, dim=1)
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

        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]
        return n, h

    def mix_dataset(self, total_samples=None):
        if total_samples is None:
            total_samples = self.cfg['data']['n_samples']
        ratios = self.cfg['generation']['ratios']

        n_lhs = int(ratios['lhs'] * total_samples)
        n_rnd = int(ratios['random_sum'] * total_samples) 
        n_single = int(ratios['single'] * total_samples)
        n_wall = int(ratios['wall'] * total_samples)
        n_sparse = int(ratios['sparse'] * total_samples)
        n_switch = int(ratios['switch'] * total_samples)

        current_sum = n_lhs + n_rnd + n_single + n_wall + n_sparse + n_switch
        n_lhs += (total_samples - current_sum)

        print(f"Generating Dataset ({total_samples} samples):")
        print(f"  - LHS:        {n_lhs}")
        print(f"  - Random Sum: {n_rnd}")
        print(f"  - Single:     {n_single}")
        print(f"  - Wall:       {n_wall}")
        print(f"  - Sparse:     {n_sparse}")
        print(f"  - Switch:     {n_switch}")

        sampler = LatinHypercube(d=2*self.n_asp)
        sample = sampler.random(n=n_lhs)
        n_lhs_data = 1.0 + torch.tensor(sample[:, :self.n_asp]).float() * 2.0
        h_lhs_data = torch.tensor(sample[:, self.n_asp:]).float() * self.max_delta

        n_rn, h_rn = self.generate_random_sums(n_rnd)
        n_si, h_si = self.generate_canonical_singles(n_single)
        n_wa, h_wa = self.generate_canonical_walls(n_wall)
        n_sp, h_sp = self.generate_sparse(n_sparse)
        n_bi, h_bi = self.generate_multistep(n_switch)

        all_n = torch.cat([n_lhs_data, n_rn, n_si, n_wa, n_sp, n_bi])
        all_h = torch.cat([h_lhs_data, h_rn, h_si, h_wa, h_sp, h_bi])

        all_h, _ = torch.sort(all_h, dim=1)
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
    n_samples = cfg['data']['n_samples']

    print(f"Generating {n_samples} mixed surface parameters...")
    all_n, all_h = gen.mix_dataset(total_samples=n_samples)

    all_n = all_n.to(device)
    all_h = all_h.to(device)

    print("Solving physics to generate Load/Area/Derivative curves...")
    phys = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)

    R = cfg['physics']['radius']
    max_d = cfg['physics']['max_delta_ratio'] * R
    n_steps = cfg['data']['n_steps']

    indentations = torch.linspace(0, max_d, n_steps).to(device).unsqueeze(0)
    t_w = torch.ones(1, cfg['physics']['n_asperities']).to(device) * 2.0 * R

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

            # CHANGED: Now receives 3 outputs, directly utilizing the analytical dF/dA
            load, area, dF_dA = phys(batch_h, batch_n, t_w, batch_ind)

            all_loads.append(load.cpu())
            all_areas.append(area.cpu())
            all_stiff.append(dF_dA.cpu())

    # X_final now contains 3 channels: [0] = Load, [1] = Area, [2] = dF/dA
    X_final = torch.stack([
        torch.cat(all_loads, dim=0),
        torch.cat(all_areas, dim=0),
        torch.cat(all_stiff, dim=0) 
    ], dim=1)

    Y_final = torch.cat([all_n.cpu(), all_h.cpu()], dim=1)

    save_path = cfg['data']['path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving dataset to: {save_path}")
    torch.save({
        "x": X_final,
        "y": Y_final
    }, save_path)

    print("--- Dataset Generation Complete ---")