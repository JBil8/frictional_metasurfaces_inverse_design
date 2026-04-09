import torch
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TargetGenerator:
    def __init__(self, phys_engine, cfg, device):
        self.phys = phys_engine
        self.device = device
        self.n_asp = cfg['physics']['n_asperities']
        self.n_steps = cfg['data']['n_steps']
        self.max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        self.R = cfg['physics']['radius']

        self.t_w = torch.ones(1, self.n_asp).to(device) * 2.0 * self.R
        self.indentations = torch.linspace(0, self.max_d, self.n_steps).unsqueeze(0).to(device)
        
        h_wall = torch.zeros(1, self.n_asp).to(device)
        n_wall = torch.ones(1, self.n_asp).to(device) * 3.0
        w_wall = self.t_w.clone()

        with torch.no_grad():
            self.p_max, self.alpha_max, self.s_max = self.phys(h_wall, n_wall, w_wall, self.indentations)

        n_cone = torch.ones(1, self.n_asp).to(device) * 1.0
        with torch.no_grad():
            self.p_min, self.alpha_min, self.s_min = self.phys(h_wall, n_cone, w_wall, self.indentations)

        print("[TargetGenerator] Loading dataset for validation sampling...")
        data_path = cfg['data']['path']
        if not os.path.exists(data_path):
            data_path = os.path.join("..", data_path)

        self.data = torch.load(data_path, map_location=device)
        self.total_samples = self.data['y'].shape[0]

        self.ranges = self._calculate_ranges(cfg['generation']['ratios'])

    def _calculate_ranges(self, ratios):
        ranges = {}
        current_idx = 0
        total = self.total_samples
        order = ['lhs', 'random_sum', 'single', 'wall', 'sparse', 'switch']

        for key in order:
            if key not in ratios:
                count = 0
                print(f"Warning: Ratio for '{key}' missing in config. Assuming 0.")
            else:
                count = int(ratios[key] * total)

            if key == 'lhs':
                others = sum([int(ratios.get(k, 0) * total) for k in order if k != 'lhs'])
                count = total - others

            start = current_idx
            end = start + count
            ranges[key] = (start, end)
            current_idx = end

        return ranges

    def get_dataset_sample(self, category="lhs", offset=0, noise_level=0.0):
        if category not in self.ranges:
            raise ValueError(f"Unknown category: {category}")

        start, end = self.ranges[category]
        idx = start + offset

        if idx >= self.total_samples:
            idx = idx % self.total_samples

        print(f"  > Fetching '{category}' sample at global index {idx}...")

        x_sample = self.data['x'][idx].unsqueeze(0).to(self.device)
        y_sample = self.data['y'][idx].unsqueeze(0).to(self.device)

        target_pressure = x_sample[:, 0, :]
        target_alpha = x_sample[:, 1, :]
        target_stiff = x_sample[:, 2, :] # CRITICAL: Extract Stiffness

        if noise_level > 0:
            noise_s = torch.randn_like(target_stiff) * noise_level * target_stiff.max()
            target_stiff += noise_s

        gt_n = y_sample[:, :self.n_asp]
        gt_h = y_sample[:, self.n_asp:]

        return target_pressure, target_alpha, target_stiff, gt_n, gt_h, f"Dataset: {category.capitalize()} (#{offset})"

    def get_consistent_linear_coulomb(self):
        n = torch.ones(1, self.n_asp).to(self.device) * 2.0 
        h_dist = torch.distributions.Exponential(rate=10.0)
        h_vals = h_dist.sample((1, self.n_asp)).to(self.device)

        h = h_vals * (0.5 * self.max_d)
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1] 

        with torch.no_grad():
            target_pressure, target_alpha, target_stiff = self.phys(h, n, self.t_w, self.indentations)

        return target_pressure, target_alpha, target_stiff, "Linear (GW Physics)"
    
    def get_synthetic_sigmoid(self):
        """
        Creates a purely synthetic target curve where P follows a sigmoid 
        relative to alpha. Note: No ground-truth parameters exist for this.
        """
        # Create a normalized step axis [0, 1] to map to your indentations
        steps = torch.linspace(0, 1, self.n_steps).to(self.device)
        
        # 1. Define Alpha (Contact Fraction)
        # Let it grow smoothly up to a maximum of 40% contact
        t_alpha = 0.4 * (steps ** 1.5) 
        
        # 2. Define Pressure (P) as a Sigmoid of Alpha
        # Target roughly 50% of the theoretical maximum pressure capacity
        P_max = self.p_max.max() * 0.5 
        
        k = 25.0       # Steepness of the switch
        alpha_0 = 0.2  # Midpoint of the switch (triggers at 20% contact)
        
        raw_sig = torch.sigmoid(k * (t_alpha - alpha_0))
        sig_init = torch.sigmoid(torch.tensor([-k * alpha_0])).to(self.device)
        
        # Shift and scale so P strictly starts at 0
        t_p = P_max * (raw_sig - sig_init) 
        
        # 3. Calculate Target Stiffness (dP/dAlpha) numerically
        t_s = torch.zeros_like(t_p)
        dp = torch.diff(t_p)
        da = torch.diff(t_alpha)
        
        # Prevent division by zero
        valid = da > 1e-8
        t_s[1:][valid] = dp[valid] / da[valid]
        t_s[0] = t_s[1] 
        
        # Add batch dimensions
        t_p = t_p.unsqueeze(0)
        t_alpha = t_alpha.unsqueeze(0)
        t_s = t_s.unsqueeze(0)
        
        return t_p, t_alpha, t_s, "Synthetic Sigmoid Switch"

    def get_consistent_saturating(self):
        # CRITICAL FIX: Limit exponent to 3.0 to match the new restricted physical limits
        n = torch.ones(1, self.n_asp).to(self.device) * 3.0
        h_vals = torch.rand(1, self.n_asp).to(self.device)
        h = h_vals * (0.25 * self.max_d)
        h, _ = torch.sort(h, dim=1)
        h = h - h[:, 0:1]

        with torch.no_grad():
            target_pressure, target_alpha, target_stiff = self.phys(h, n, self.t_w, self.indentations)

        return target_pressure, target_alpha, target_stiff, "Saturating (Bounded Flat Punches)"

    def get_consistent_bilinear(self):
        n = torch.ones(1, self.n_asp).to(self.device) * 3.0
        
        # Determine half of the asperities dynamically
        half_n = self.n_asp // 2
        rest_n = self.n_asp - half_n
        
        h1 = torch.zeros(1, half_n).to(self.device)
        h2 = torch.ones(1, rest_n).to(self.device) * (0.4 * self.max_d)

        h = torch.cat([h1, h2], dim=1)

        with torch.no_grad():
            target_pressure, target_alpha, target_stiff = self.phys(h, n, self.t_w, self.indentations)

        return target_pressure, target_alpha, target_stiff, "Bilinear (Gap Physics)"

    def get_custom_sample(self, idx, label="Custom"):
        x_sample = self.data['x'][idx].unsqueeze(0).to(self.device)
        y_sample = self.data['y'][idx].unsqueeze(0).to(self.device)

        target_pressure = x_sample[:, 0, :]
        target_alpha = x_sample[:, 1, :]
        target_stiff = x_sample[:, 2, :] # CRITICAL: Extract Stiffness
        
        gt_n = y_sample[:, :self.n_asp]
        gt_h = y_sample[:, self.n_asp:]

        return target_pressure, target_alpha, target_stiff, gt_n, gt_h, f"Dataset: {label.capitalize()} (#{idx})"