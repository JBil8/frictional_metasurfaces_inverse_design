import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.normalization import get_theoretical_limits

from utils.config import load_config
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer

def evaluate_model_performance():
    # 1. Setup
    cfg = load_config("config.yaml")
    
    # Check for CUDA availability
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Running evaluation on: {device_str.upper()}")
    
    # 2. Load Data
    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'], map_location=device)
    
    X = data["x"] # (N, 3, Steps)
    Y = data["y"] # (N, Params)
    
    # --- CRITICAL FIX: Deterministic Split ---
    # We must replicate the exact split used in training to ensure
    # we are testing on the correct 'unseen' data.
    
    dataset = TensorDataset(X, Y)
    total_len = len(dataset)
    
    # Ratios must match your training script (e.g. 0.8, 0.1, 0.1)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    # Fixed seed for reproducibility (Must match training seed!)
    generator = torch.Generator().manual_seed(42)
    
    _, _, test_ds = random_split(
        dataset, 
        [train_len, val_len, test_len], 
        generator=generator
    )
    
    print(f"Test Set Size: {len(test_ds)} samples (Randomly sampled, consistent with Training)")
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)
    
    # 3. Calculate Normalization Factors
    limits = get_theoretical_limits(cfg, device)
    MAX_L = limits['max_load']
    MAX_A = limits['max_area']
    MAX_S = limits['max_stiff']
    
    # 4. Load Model
    model = SurfaceInverseModel(cfg).to(device)
    
    print("Loading model weights...")
    model_name = cfg['model']['name']
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Physics Layer for Reconstruction
    physics = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
    indentations = torch.linspace(0, max_d, cfg['data']['n_steps']).unsqueeze(0).to(device)

    # 5. Accumulate Errors
    mse_load = 0.0
    mse_area = 0.0
    mse_params = 0.0
    total_samples = 0
    
    criterion = nn.MSELoss(reduction='sum')
    
    print(f"Starting evaluation loop...")
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            batch_size = bx.size(0)
            
            # --- Normalize Input on the fly ---
            # (Because random_split gave us raw tensors from the original X)
            bx_norm = bx.clone()
            bx_norm[:, 0, :] /= MAX_L
            bx_norm[:, 1, :] /= MAX_A
            bx_norm[:, 2, :] /= MAX_S
            
            # Predict (Pass Normalized Input)
            p_n, p_h = model(bx_norm)
            
            # Reconstruct Physics
            p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
            batch_ind = indentations.repeat(batch_size, 1)
            rec_l, rec_a = physics(p_h, p_n, p_w, batch_ind)
            
            # Normalize Reconstruction (to match input scale for loss)
            rec_l /= MAX_L
            rec_a /= MAX_A
            
            # Calculate Errors (Compare against normalized inputs)
            loss_l = criterion(rec_l, bx_norm[:, 0, :])
            loss_a = criterion(rec_a, bx_norm[:, 1, :])
            
            # Parameter Error
            pred_params = torch.cat([p_n, p_h], dim=1)
            loss_p = criterion(pred_params, by)
            
            mse_load += loss_l.item()
            mse_area += loss_a.item()
            mse_params += loss_p.item()
            total_samples += batch_size

    # 6. Final Calculation
    n_steps = cfg['data']['n_steps']
    n_params = Y.shape[1]
    
    avg_mse_load = mse_load / (total_samples * n_steps)
    avg_mse_area = mse_area / (total_samples * n_steps)
    avg_mse_params = mse_params / (total_samples * n_params)
    
    print("="*40)
    print(f"FINAL TEST RESULTS (Normalized Units)")
    print("="*40)
    print(f"Load Reconstruction MSE : {avg_mse_load:.2e}")
    print(f"Area Reconstruction MSE : {avg_mse_area:.2e}")
    print(f"Parameter Prediction MSE: {avg_mse_params:.2e}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model_performance()