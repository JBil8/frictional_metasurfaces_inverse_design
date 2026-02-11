import torch
import sys
import os
import numpy as np

# Adjust path to import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TargetGenerator:
    def __init__(self, phys_engine, cfg, device):
        self.phys = phys_engine
        self.device = device
        self.n_asp = cfg['physics']['n_asperities']
        self.n_steps = cfg['data']['n_steps']
        self.max_d = cfg['physics']['max_delta_ratio'] * \
            cfg['physics']['radius']
        self.R = cfg['physics']['radius']

        # Standard width
        self.t_w = torch.ones(1, self.n_asp).to(device) * 2.0 * self.R
        self.indentations = torch.linspace(
            0, self.max_d, self.n_steps).unsqueeze(0).to(device)
        h_wall = torch.zeros(1, self.n_asp).to(device)
        n_wall = torch.ones(1, self.n_asp).to(device) * 8.0
        w_wall = self.t_w.clone()

        # 2. Solve Physics
        with torch.no_grad():
            self.l_max, self.a_max = self.phys(
                h_wall, n_wall, w_wall, self.indentations)

        # 3. Also calculate the "Softest" limit (Cone, n=1) for lower bounds
        n_cone = torch.ones(1, self.n_asp).to(device) * 1.0
        with torch.no_grad():
            self.l_min, self.a_min = self.phys(
                h_wall, n_cone, w_wall, self.indentations)
        # Load the Dataset for "Real Sample" validation
        print("[TargetGenerator] Loading dataset for validation sampling...")
        data_path = cfg['data']['path']
        if not os.path.exists(data_path):
            data_path = os.path.join("..", data_path)

        self.data = torch.load(data_path, map_location=device)
        self.total_samples = self.data['y'].shape[0]

        # Define Category Indices based on your generation ratios
        self.ranges = self._calculate_ranges(cfg['generation']['ratios'])

    def _calculate_ranges(self, ratios):
        """
        Converts config ratios into absolute start/end indices.
        """
        ranges = {}
        current_idx = 0
        total = self.total_samples

        # CRITICAL: This order must match 'mix_dataset' in surface_generator.py
        order = ['lhs', 'random_sum', 'single', 'wall', 'sparse', 'switch']

        for key in order:
            # Safety: If config is missing the new key, assume 0
            if key not in ratios:
                count = 0
                print(
                    f"Warning: Ratio for '{key}' missing in config. Assuming 0.")
            else:
                count = int(ratios[key] * total)

            # Rounding fix for LHS
            if key == 'lhs':
                others = sum([int(ratios.get(k, 0) * total)
                             for k in order if k != 'lhs'])
                count = total - others

            start = current_idx
            end = start + count
            ranges[key] = (start, end)
            current_idx = end

        return ranges

    def get_dataset_sample(self, category="lhs", offset=0, noise_level=0.0):
        """
        Fetches a real sample from the dataset.
        Args:
            category: One of ['lhs', 'single', 'wall', 'sparse', 'switch']
            offset: Index offset (e.g., 0 for the first sample of that type)
            noise_level: Add Gaussian noise to the Load/Area curves inputs?
        """

        if category not in self.ranges:
            raise ValueError(f"Unknown category: {category}")

        start, end = self.ranges[category]
        idx = start + offset

        # Safety check
        if idx >= self.total_samples:
            print(f"Warning: Index {idx} out of bounds. wrapping around.")
            idx = idx % self.total_samples

        print(f"  > Fetching '{category}' sample at global index {idx}...")

        # Extract X (Curves) and Y (Params)
        # X shape: [1, 3, Steps] -> Load, Area, Stiffness
        # Y shape: [1, 32] -> n, h
        x_sample = self.data['x'][idx].unsqueeze(0).to(self.device)
        y_sample = self.data['y'][idx].unsqueeze(0).to(self.device)

        target_load = x_sample[:, 0, :]
        target_area = x_sample[:, 1, :]

        # Add Noise (Robustness Test)
        if noise_level > 0:
            noise_l = torch.randn_like(
                target_load) * noise_level * target_load.max()
            noise_a = torch.randn_like(
                target_area) * noise_level * target_area.max()
            target_load += noise_l
            target_area += noise_a

        gt_n = y_sample[:, :self.n_asp]
        gt_h = y_sample[:, self.n_asp:]

        return target_load, target_area, gt_n, gt_h, f"Dataset: {category.capitalize()} (#{offset})"

    def get_consistent_linear_coulomb(self):
        """
        Generates a Linear Target (A ~ P) by simulating a PHYSICALLY VALID
        Exponential Distribution of heights (Greenwood-Williamson).
        """
        # 1. Generate Parameters explicitly for the "Unseen" Physics
        # Exponential distribution = Linear Contact Law
        n = torch.ones(1, self.n_asp).to(self.device) * 2.0  # Spheres (Hertz)

        # Exponential heights (The key to linearity)
        # We construct this manually to ensure it's "perfectly" exponential
        # independent of the random generator
        h_dist = torch.distributions.Exponential(rate=10.0)
        h_vals = h_dist.sample((1, self.n_asp)).to(self.device)

        # Scale to physical range
        h = h_vals * (0.5 * self.max_d)
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]  # Normalize

        # 2. SOLVE PHYSICS to get the True Consistent Curve
        with torch.no_grad():
            target_load, target_area = self.phys(
                h, n, self.t_w, self.indentations)

        return target_load, target_area, "Linear (GW Physics)"

    def get_consistent_saturating(self):
        """
        Generates a True Saturating curve (Roughness Flattening).
        Strategy: Use 'Flat Punches' (n=8) with a BOUNDED height distribution.
        """
        # 1. Maximize Exponent (Flat Punch behavior)
        # For a flat punch, Area is constant with depth (A ~ d^0), causing perfect saturation.
        n = torch.ones(1, self.n_asp).to(self.device) * 6.0

        # 2. Bounded Height Distribution (Uniform)
        # Gaussian has tails (infinite heights). Uniform has a hard cutoff.
        # We confine all asperities to the first 25% of the max depth.
        # Once indentation > 0.25, Area will effectively stop growing.
        h_vals = torch.rand(1, self.n_asp).to(self.device)
        h = h_vals * (0.25 * self.max_d)

        # Sort & Normalize
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]

        # Solve Physics
        with torch.no_grad():
            target_load, target_area = self.phys(
                h, n, self.t_w, self.indentations)

        return target_load, target_area, "Saturating (Bounded Flat Punches)"

    def get_consistent_bilinear(self):
        """
        Generates a Bilinear Target by mixing two distinct height groups.
        """
        n = torch.ones(1, self.n_asp).to(self.device) * 3.0

        # Group 1: At 0 (Touching)
        h1 = torch.zeros(1, 8).to(self.device)
        # Group 2: At Gap (Delayed)
        h2 = torch.ones(1, 8).to(self.device) * (0.4 * self.max_d)

        h = torch.cat([h1, h2], dim=1)

        # Solve Physics
        with torch.no_grad():
            target_load, target_area = self.phys(
                h, n, self.t_w, self.indentations)

        return target_load, target_area, "Bilinear (Gap Physics)"

    def get_bilinear_transition(self):
        """
        Two linear regimes: Soft start -> Stiff finish.
        """
        # Transition happens at moderate loads
        scale_factor = 0.25
        target_load = self.l_max * scale_factor
        norm_P = target_load / target_load.max()

        max_A = self.a_max.max().item() * 0.12  # Scale area expectations too

        knee_load = 0.4

        s1 = 0.8 * max_A
        s2 = 0.2 * max_A

        target_area = torch.zeros_like(target_load)
        mask_low = norm_P <= knee_load
        mask_high = ~mask_low

        target_area[mask_low] = norm_P[mask_low] * s1
        A_knee = knee_load * s1
        target_area[mask_high] = A_knee + (norm_P[mask_high] - knee_load) * s2

        return target_load, target_area, "Bilinear (Soft-to-Stiff)"

    def get_custom_sample(self, idx, label="Custom"):
        """Fetches a specific index directly."""
        x_sample = self.data['x'][idx].unsqueeze(0).to(self.device)
        y_sample = self.data['y'][idx].unsqueeze(0).to(self.device)

        target_load = x_sample[:, 0, :]
        target_area = x_sample[:, 1, :]
        gt_n = y_sample[:, :self.n_asp]
        gt_h = y_sample[:, self.n_asp:]

        return target_load, target_area, gt_n, gt_h, f"Dataset: {label.capitalize()} (#{idx})"
