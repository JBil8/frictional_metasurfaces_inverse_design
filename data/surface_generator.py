import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube
from physics.differentiable import AxisymmetricContactLayer
from utils.config import load_config

class SurfaceDataGenerator:
    def __init__(self, config):
        self.cfg = config
        self.n = config['physics']['n_asperities']
        self.n_steps = config['data']['n_steps']
        # Access nested values clearly
        self.max_delta = config['physics']['max_delta_ratio'] * config['physics']['radius']# Physical max indentation
        self.R = config['physics']['radius']
        self.physics = AxisymmetricContactLayer(E_star=config['physics']['E_star'])
        
        # Fixed grid for all samples
        self.indentations = torch.linspace(0, self.max_delta, self.n_steps)  

    def generate_dataset(self, n_samples=10000):
        """
        Generates a dataset using Latin Hypercube Sampling.
        Returns:
            inputs: (N, 2, Steps) -> [Load, Area] curves (The NN Input)
            params: (N, n_params) -> [Exponents, Height_Diffs] (The GT Output)
        """
        print(f"Generating {n_samples} samples with LHS...")
        
        # Parameter Sampling (LHS)
        # We have 2 types of parameters to sample:
        # A. Exponents (n): Range [1.0, 8.0]
        # B. Heights (z): Range [0, max_delta]
        
        d_dims = self.n + self.n  # Sampling heights (n) and exponents (n)
        sampler = LatinHypercube(d=d_dims)
        sample = sampler.random(n=n_samples) # [N, 2*n] in [0,1] range

        # Scale Exponents: [0,1] -> [1.0, 8.0]
        exponents = 1.0 + sample[:, :self.n] * 7.0 
        
        # Scale Heights: [0,1] -> [0, max_delta]
        # We sample raw heights first, then sort them to get offsets
        raw_heights = sample[:, self.n:] * self.max_delta
        
        # Sort heights so h1 < h2 < ...
        # This removes permutation symmetry and ensures reliable training
        sorted_heights = np.sort(raw_heights, axis=1)
        
        # Normalize so the first asperity is always at z=0 (Touch point)
        # We only care about relative differences
        offsets = sorted_heights - sorted_heights[:, 0:1] # [N, n]
        
        # Convert to Tensors
        exponents_t = torch.tensor(exponents, dtype=torch.float32)
        offsets_t = torch.tensor(offsets, dtype=torch.float32)
        widths_t = torch.ones_like(exponents_t) * (2.0 * self.R) # Fixed Width for now
        
        # Prepare batch of indentations: [N, Steps]
        indent_batch = self.indentations.unsqueeze(0).repeat(n_samples, 1)

        # --- 4. Run Physics Engine (Vectorized) ---
        # We process in chunks to avoid GPU/RAM OOM if N is huge
        batch_size = 1000
        loads_list = []
        areas_list = []
        
        print("Running Physics Engine...")
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                # Slice batches
                h_b = offsets_t[i:i+batch_size]
                n_b = exponents_t[i:i+batch_size]
                w_b = widths_t[i:i+batch_size]
                d_b = indent_batch[i:i+batch_size]
                
                # Forward Pass
                l, a = self.physics(h_b, n_b, w_b, d_b)
                
                loads_list.append(l)
                areas_list.append(a)

        total_load = torch.cat(loads_list, dim=0)
        total_area = torch.cat(areas_list, dim=0)

        # --- 5. Data Packaging ---
        # Input to NN: The curves
        # Stiffness = d(Load) / d(Indentation Step)
        # We use torch.diff to calculate difference between steps.
        # We prepend a zero to keep the sequence length the same (n_steps).
        stiffness = torch.diff(total_load, dim=1, prepend=torch.zeros(n_samples, 1))
        
        # --- 5. Data Packaging ---
        # Input to NN: Now 3 Channels [Load, Area, Stiffness]
        curves = torch.stack([total_load, total_area, stiffness], dim=1)
        
        params = torch.cat([exponents_t, offsets_t], dim=1)
        
        print(f"Dataset Generated. Curves: {curves.shape}, Params: {params.shape}")
        return curves, params

if __name__ == "__main__":
    cfg = load_config()
    gen = SurfaceDataGenerator(cfg)
    # Access data params
    curves, params = gen.generate_dataset(n_samples=cfg['data']['n_samples'])
    # Save for training
    torch.save({"x": curves, "y": params}, "data/dataset_16_asp.pt")