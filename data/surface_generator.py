from physics.differentiable import AxisymmetricContactLayer
from utils.interpolation import batched_interp1d
from utils.seeding import set_seed
from utils.config import load_config
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats.qmc import LatinHypercube

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SurfaceGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_asp = cfg['physics']['n_asperities']
        self.max_delta = cfg['physics']['max_delta_ratio'] * \
            cfg['physics']['radius']
        self.gamma_min = cfg['physics']['gamma_min']
        self.gamma_max = cfg['physics']['gamma_max']

    def get_base_batch(self, n_samples):
        return torch.zeros(n_samples, self.n_asp), torch.zeros(n_samples, self.n_asp)

    def generate_exiled_contacts(self, n_samples):
        print(
            f"  > Generating {n_samples} Exiled Contacts (Learning to hide asperities)...")
        gamma, h = self.get_base_batch(n_samples)
        gamma = self.gamma_min + \
            torch.rand(n_samples, self.n_asp) * \
            (self.gamma_max - self.gamma_min)

        for i in range(n_samples):
            # Only 1 to 3 asperities actually make contact
            n_active = np.random.randint(1, 4)
            h_active = torch.rand(n_active) * (0.5 * self.max_delta)
            # The rest are exiled beyond max_delta so they NEVER touch
            h_exiled = (1.1 + torch.rand(self.n_asp - n_active)) * \
                self.max_delta
            combined = torch.cat([h_active, h_exiled])
            h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        gamma = torch.gather(gamma, 1, sorted_idx)
        h = h - h[:, 0:1]
        return gamma, h

    def generate_bimodal_extremes(self, n_samples):
        print(f"  > Generating {n_samples} Bimodal Extremes (Violent stiffness cliffs)...")
        gamma, h = self.get_base_batch(n_samples)
        
        # Calculate the dynamic range
        gamma_range = self.gamma_max - self.gamma_min

        for i in range(n_samples):
            split = self.n_asp // 2

            # Soft asperities cluster near the lower physical limit (e.g. bottom 20% of range)
            gamma_soft = self.gamma_min + torch.rand(split) * (0.2 * gamma_range)
            h_soft = torch.rand(split) * (0.2 * self.max_delta)

            # Blunt asperities cluster near the upper physical limit (e.g. top 20% of range)
            n_blunt = self.gamma_max - torch.rand(self.n_asp - split) * (0.2 * gamma_range)
            h_blunt = (0.5 + torch.rand(self.n_asp - split) * 0.4) * self.max_delta

            combined_h = torch.cat([h_soft, h_blunt])
            combined_n = torch.cat([gamma_soft, n_blunt])

            perm = torch.randperm(self.n_asp)
            h[i] = combined_h[perm]
            gamma[i] = combined_n[perm]

        h, sorted_idx = torch.sort(h, dim=1)
        gamma = torch.gather(gamma, 1, sorted_idx)
        h = h - h[:, 0:1]
        return gamma, h

    def generate_canonical_walls(self, n_samples):
        print(f"  > Generating {n_samples} Canonical Walls ...")
        gamma, h = self.get_base_batch(n_samples)
        gamma = self.gamma_min + \
            torch.rand(n_samples, self.n_asp) * \
            (self.gamma_max - self.gamma_min)

        for i in range(n_samples):
            mode = np.random.rand()
            if mode < 0.5:
                # All 9 touch almost immediately
                roughness = torch.rand(self.n_asp) * (0.01 * self.max_delta)
                h[i] = roughness
            else:
                n_active = np.random.randint(4, self.n_asp)
                h_active = torch.rand(n_active) * (0.01 * self.max_delta)
                h_inactive = (
                    0.5 + 0.5 * torch.rand(self.n_asp - n_active)) * self.max_delta
                combined = torch.cat([h_active, h_inactive])
                h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        gamma = torch.gather(gamma, 1, sorted_idx)
        h = h - h[:, 0:1]
        return gamma, h

    def generate_sparse(self, n_samples):
        print(f"  > Generating {n_samples} Sparse (Sequential) samples...")
        gamma, h = self.get_base_batch(n_samples)
        gamma = self.gamma_min + \
            torch.rand(n_samples, self.n_asp) * \
            (self.gamma_max - self.gamma_min)

        for i in range(n_samples):
            n_active = np.random.randint(2, max(3, self.n_asp // 2))
            h_active = torch.rand(n_active) * (0.8 * self.max_delta)
            h_inactive = (0.8 + 0.2 * torch.rand(self.n_asp -
                          n_active)) * self.max_delta
            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        gamma = torch.gather(gamma, 1, sorted_idx)
        h = h - h[:, 0:1]
        return gamma, h

    def generate_random_sums(self, n_samples):
        print(f"  > Generating {n_samples} Random Sums ...")
        gamma, h = self.get_base_batch(n_samples)
        gamma = self.gamma_min + \
            torch.rand(n_samples, self.n_asp) * \
            (self.gamma_max - self.gamma_min)

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
                raw_exp = torch.distributions.Exponential(
                    rate=3.0).sample((n_active,))
                h_active = raw_exp * (0.3 * self.max_delta)

            h_active = torch.clamp(h_active, 0, self.max_delta)
            h_inactive = (1.0 + 0.2 * torch.rand(self.n_asp -
                          n_active)) * self.max_delta

            combined = torch.cat([h_active, h_inactive])
            h[i] = combined[torch.randperm(self.n_asp)]

        h, sorted_idx = torch.sort(h, dim=1)
        gamma = torch.gather(gamma, 1, sorted_idx)
        h = h - h[:, 0:1]
        return gamma, h

    def mix_dataset(self, total_samples):
        # Extract ratios from config
        ratios = self.cfg['generation']['ratios']

        n_exiled = int(ratios['exiled'] * total_samples)
        n_bimodal = int(ratios['bimodal'] * total_samples)
        n_wall = int(ratios['wall'] * total_samples)
        n_sparse = int(ratios['sparse'] * total_samples)
        n_rnd = int(ratios['random_sum'] * total_samples)

        # LHS absorbs the remainder to guarantee exactly total_samples are generated
        n_lhs = total_samples - \
            (n_exiled + n_bimodal + n_wall + n_sparse + n_rnd)

        print("--- Mixing Dataset Sub-Domains ---")
        sampler = LatinHypercube(d=2*self.n_asp)
        sample = sampler.random(n=n_lhs)
        n_lhs_data = self.gamma_min + \
            torch.tensor(sample[:, :self.n_asp]).float() * \
            (self.gamma_max - self.gamma_min)
        h_lhs_data = torch.tensor(
            sample[:, self.n_asp:]).float() * self.max_delta

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
    set_seed()

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
    # 1. ESTABLISH THE GLOBAL PHYSICAL BOUNDING BOX (Optional but good for reference)
    # ---------------------------------------------------------
    print("Calculating Global Maximums for reference...")
    n_ceiling = torch.ones(1, n_asp).to(device) * gen.gamma_max
    h_ceiling = torch.zeros(1, n_asp).to(device)

    with torch.no_grad():
        P_bound, _, _ = phys(h_ceiling, n_ceiling, t_w,
                             indentations, k_steepness=1e5)
        global_P_max = P_bound[0, -1].item()

    print(f"Global P*_max established at: {global_P_max:.6f}")

    # ---------------------------------------------------------
    # 2. RUN PHYSICS AND INTERPOLATE ONTO NORMALIZED [0, 1] GRID
    # ---------------------------------------------------------
    print("Solving physics and interpolating to Normalized P-hat grid...")
    batch_size = cfg['training']['batch_size']
    dataset = TensorDataset(all_n, all_h)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_alpha_hat = []
    all_stiff_hat = []
    all_scalars = []

    # The universal x-axis for the network (0.0 to 1.0)
    p_hat_grid = torch.linspace(0, 1.0, n_steps, device=device)

    with torch.no_grad():
        for batch_n, batch_h in loader:
            batch_n = batch_n.to(device)
            batch_h = batch_h.to(device)
            current_batch = batch_n.shape[0]
            batch_ind = indentations.repeat(current_batch, 1)
            batch_tw = t_w.repeat(current_batch, 1)

            # Generate native curves in displacement space
            P_native, alpha_native, stiff_native = phys(
                batch_h, batch_n, batch_tw, batch_ind, k_steepness=1e5)

            # 1. Extract absolute maximums at maximum indentation
            # Slice with [-1:] to keep the dimension (Batch, 1) for easy broadcasting
            p_max = P_native[:, -1:]
            a_max = alpha_native[:, -1:]

            # Prevent division by zero mathematically using clamp
            p_max_safe = torch.clamp(p_max, min=1e-12)
            a_max_safe = torch.clamp(a_max, min=1e-12)

            # 2. Normalize arrays purely in PyTorch (Broadcasting handles the Batch dimension)
            P_hat = P_native / p_max_safe
            alpha_hat = alpha_native / a_max_safe

            # Normalized Stiffness: d(P_hat)/d(alpha_hat) = S * (a_max / p_max)
            stiff_hat = stiff_native * (a_max_safe / p_max_safe)

            # 3. Interpolate onto the universal 0-to-1 grid using your custom function
            # Since P_hat and p_hat_grid both max out at 1.0, pad_value won't trigger, but 0.0 or 1.0 is safe.
            batch_alpha_interp = batched_interp1d(
                p_hat_grid, P_hat, alpha_hat, pad_value=1.0)
            batch_stiff_interp = batched_interp1d(
                p_hat_grid, P_hat, stiff_hat, pad_value=0.0)

            # Store scalars (Concatenate along dim 1 to make shape: [Batch, 2])
            batch_scalars = torch.cat([p_max, a_max], dim=1)

            # Move to CPU only at the very end to free up GPU VRAM for the next batch
            all_alpha_hat.append(batch_alpha_interp.cpu())
            all_stiff_hat.append(batch_stiff_interp.cpu())
            all_scalars.append(batch_scalars.cpu())

    # ---------------------------------------------------------
    # 3. ASSEMBLE AND SAVE
    # ---------------------------------------------------------
    # X_arrays Shape: (Total_Samples, 2 channels, 512 steps)
    X_arrays = torch.stack([
        torch.cat(all_alpha_hat, dim=0),
        torch.cat(all_stiff_hat, dim=0)
    ], dim=1)

    # X_scalars Shape: (Total_Samples, 2)
    X_scalars = torch.cat(all_scalars, dim=0)

    # Y_final Shape: (Total_Samples, 18)
    Y_final = torch.cat([all_n.cpu(), all_h.cpu()], dim=1)

    save_path = cfg['data']['path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        "x_arrays": X_arrays,
        "x_scalars": X_scalars,
        "y": Y_final,
        "p_star_max_global": global_P_max
    }, save_path)

    print("--- Dataset Generation Complete ---")
