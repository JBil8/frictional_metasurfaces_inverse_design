import torch
import numpy as np
import sys
import os

# Adjust path to import from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TargetGenerator:
    """
    Generates physically grounded target curves.
    Crucially, it probes the physics engine first to ensure targets are feasible.
    """
    def __init__(self, phys_engine, cfg, device):
        self.phys = phys_engine
        self.device = device
        self.n_asp = cfg['physics']['n_asperities']
        self.n_steps = cfg['data']['n_steps']
        self.max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        self.R = cfg['physics']['radius']
        
        # Standard width (assumes constant radius for this specific paper/project)
        self.t_w = torch.ones(1, self.n_asp).to(device) * 2.0 * self.R
        self.indentations = torch.linspace(0, self.max_d, self.n_steps).unsqueeze(0).to(device)
        
        # Pre-calculate Physical Envelope (Min/Max possible Area)
        self.l_min, self.a_min, self.l_max, self.a_max = self._probe_envelope()

    def _probe_envelope(self):
        """Finds the stiffest (Punch) and softest (Cone) limits of the system."""
        print("  [TargetGenerator] Probing physical limits of the system...")
        
        # 1. Softest Limit: Single Sharp Cone (n=1)
        # We set one asperity to h=0, others to h=max_dist
        h_soft = torch.zeros(1, self.n_asp).to(self.device) + self.max_d
        h_soft[0, 0] = 0.0
        n_soft = torch.ones(1, self.n_asp).to(self.device) * 1.0
        
        # 2. Stiffest Limit: All Flat Punches (n=8)
        # All asperities touching (h=0) with max exponent
        h_stiff = torch.zeros(1, self.n_asp).to(self.device)
        n_stiff = torch.ones(1, self.n_asp).to(self.device) * 8.0
        
        with torch.no_grad():
            l_min, a_min = self.phys(h_soft, n_soft, self.t_w, self.indentations)
            l_max, a_max = self.phys(h_stiff, n_stiff, self.t_w, self.indentations)
            
        return l_min, a_min, l_max, a_max

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