import torch
import numpy as np
import sys
import os

# Adjust path to import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import os
import sys

# Adjust path to import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TargetGenerator:
    def __init__(self, phys_engine, cfg, device):
        self.phys = phys_engine
        self.device = device
        self.n_asp = cfg['physics']['n_asperities']
        self.n_steps = cfg['data']['n_steps']
        self.max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        self.R = cfg['physics']['radius']
        
        # Standard width
        self.t_w = torch.ones(1, self.n_asp).to(device) * 2.0 * self.R
        self.indentations = torch.linspace(0, self.max_d, self.n_steps).unsqueeze(0).to(device)
        
        # Load the Dataset for "Real Sample" validation
        print("[TargetGenerator] Loading dataset for validation sampling...")
        data_path = cfg['data']['path']
        if not os.path.exists(data_path): data_path = os.path.join("..", data_path)
        
        self.data = torch.load(data_path, map_location=device)
        self.total_samples = self.data['y'].shape[0]
        
        # Define Category Indices based on your generation ratios
        # Ratios: 50% LHS, 10% Single, 5% Wall, 15% Sparse, 20% Bimodal
        self.indices = {
            "lhs":      0,
            "single":   int(0.50 * self.total_samples),
            "wall":     int(0.60 * self.total_samples),
            "sparse":   int(0.65 * self.total_samples),
            "switch":   int(0.80 * self.total_samples) # Bimodal starts at 80%
        }

    def get_dataset_sample(self, category="lhs", offset=0, noise_level=0.0):
        """
        Fetches a real sample from the dataset.
        Args:
            category: One of ['lhs', 'single', 'wall', 'sparse', 'switch']
            offset: Index offset (e.g., 0 for the first sample of that type)
            noise_level: Add Gaussian noise to the Load/Area curves inputs?
        """
        start_idx = self.indices.get(category, 0)
        idx = start_idx + offset
        
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
            noise_l = torch.randn_like(target_load) * noise_level * target_load.max()
            noise_a = torch.randn_like(target_area) * noise_level * target_area.max()
            target_load += noise_l
            target_area += noise_a
            
        gt_n = y_sample[:, :self.n_asp]
        gt_h = y_sample[:, self.n_asp:]
        
        return target_load, target_area, gt_n, gt_h, f"Dataset: {category.capitalize()} (#{offset})"

    def get_power_law(self, exponent=1.5):
        """Generates a standard Hertzian-like power law target."""
        # We define P(d) and A(d) based on the "Max" envelope but scaled down
        target_load = self.l_max * 0.5 # Target 50% of max load capacity
        
        # Theoretical Hertz: A ~ P^(2/3) (for n=2)
        # General Power Law: A ~ P^(2/(n+1))
        norm_load = target_load / target_load.max()
        
        # Scale Area based on the system's max area
        target_area = self.a_max.max() * 0.5 * (norm_load ** (2.0 / (exponent + 1.0)))
        
        return target_load, target_area, f"Power Law (Exponent {exponent})"

    def get_friction_switch(self):
        """Generates a feasible Bi-Modal Switch."""
        # Use the Envelope to guarantee feasibility!
        target_load = self.l_max.clone() # Use full load capacity
        
        # Create Sigmoid Transition
        # This creates a smooth step from 0 to 1 over the duration
        s_curve = torch.sigmoid(torch.linspace(-10, 10, self.n_steps)).to(self.device).unsqueeze(0)
        
        # Phase 1: Slip (Low Friction)
        # Target slightly more area than a single cone (e.g. 1.5 cones)
        curve_slip = self.a_min * 1.5
        
        # Phase 2: Lock (High Friction)
        # Target 90% of the maximum possible area (Punch behavior)
        curve_lock = self.a_max * 0.9
        
        # Blend them: (1-s)*Slip + s*Lock
        target_area = (1 - s_curve) * curve_slip + s_curve * curve_lock
        
        return target_load, target_area, "Friction Switch (Bimodal)"

    def get_step_contact(self, n_steps=3):
        """Generates a 'Staircase' target (Discrete jumps)."""
        target_load = self.l_max * 0.8
        target_area = torch.zeros_like(target_load)
        
        max_a = self.a_max.max().item() * 0.8
        
        # Create discrete steps
        for i in range(self.n_steps):
            # Simple logic to create stepped area
            step_idx = int((i / self.n_steps) * n_steps) + 1
            target_area[0, i] = (step_idx / n_steps) * max_a
            
        return target_load, target_area, "Step Function"
    
    def get_custom_sample(self, idx, label="Custom"):
        """Fetches a specific index directly."""
        x_sample = self.data['x'][idx].unsqueeze(0).to(self.device)
        y_sample = self.data['y'][idx].unsqueeze(0).to(self.device)
        
        target_load = x_sample[:, 0, :]
        target_area = x_sample[:, 1, :]
        gt_n = y_sample[:, :self.n_asp]
        gt_h = y_sample[:, self.n_asp:]
        
        return target_load, target_area, gt_n, gt_h, f"Dataset: {label.capitalize()} (#{idx})"