import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config
from utils.normalization import get_theoretical_limits
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer

def evaluate_model_performance():
    cfg = load_config("config.yaml")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Running evaluation on: {device_str.upper()}")

    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'], map_location=device)

    # X is now [Pressure, Alpha, dP/dAlpha]
    X = data["x"]  
    Y = data["y"]  

    dataset = TensorDataset(X, Y)
    total_len = len(dataset)

    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    # Fixed seed for reproducibility (Must match training seed!)
    generator = torch.Generator().manual_seed(42)

    _, _, test_ds = random_split(
        dataset, [train_len, val_len, test_len],
        generator=generator
    )

    print(f"Test Set Size: {len(test_ds)} samples")
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)

    # 3. Calculate Normalization Factors (Intensive)
    limits = get_theoretical_limits(cfg, device)
    MAX_P = limits['max_pressure']
    MAX_ALPHA = limits['max_alpha']
    MAX_S = limits['max_stiff']

    # 4. Load Model
    model = SurfaceInverseModel(cfg).to(device)

    print("Loading model weights...")
    state_dict = torch.load(cfg['model']['name'], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Physics Layer for Reconstruction (Pass full cfg)
    physics = AxisymmetricContactLayer(cfg=cfg).to(device)
    max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
    indentations = torch.linspace(0, max_d, cfg['data']['n_steps']).unsqueeze(0).to(device)

    # 5. Accumulate Errors
    mse_pressure = 0.0
    mse_alpha = 0.0
    mse_stiff = 0.0
    mse_params = 0.0
    total_samples = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            batch_size = bx.size(0)

            # --- Normalize Input on the fly ---
            bx_norm = bx.clone()
            bx_norm[:, 0, :] /= MAX_P
            bx_norm[:, 1, :] /= MAX_ALPHA
            bx_norm[:, 2, :] /= MAX_S

            # Predict (Pass ONLY Normalized Stiffness: Channel 2)
            p_n, p_h = model(bx_norm[:, 2:3, :])

            # Reconstruct Physics
            p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
            batch_ind = indentations.repeat(batch_size, 1)
            
            # Unpack all 3 intensive outputs
            rec_p, rec_alpha, rec_s = physics(p_h, p_n, p_w, batch_ind)

            # Normalize Reconstruction (to match input scale for loss)
            rec_p /= MAX_P
            rec_alpha /= MAX_ALPHA
            rec_s /= MAX_S

            # Calculate Errors (Compare against normalized inputs)
            loss_p = criterion(rec_p, bx_norm[:, 0, :])
            loss_a = criterion(rec_alpha, bx_norm[:, 1, :])
            loss_s = criterion(rec_s, bx_norm[:, 2, :]) 
            
            # Parameter Error
            pred_params = torch.cat([p_n, p_h], dim=1)
            loss_params = criterion(pred_params, by)

            mse_pressure += loss_p.item()
            mse_alpha += loss_a.item()
            mse_stiff += loss_s.item()
            mse_params += loss_params.item()
            total_samples += batch_size

    # 6. Final Calculation
    n_steps = cfg['data']['n_steps']
    n_params = Y.shape[1]

    avg_mse_pressure = mse_pressure / (total_samples * n_steps)
    avg_mse_alpha = mse_alpha / (total_samples * n_steps)
    avg_mse_stiff = mse_stiff / (total_samples * n_steps)
    avg_mse_params = mse_params / (total_samples * n_params)

    print("="*50)
    print(f"FINAL TEST RESULTS (Intensive Normalized Units)")
    print("="*50)
    print(f"Stiffness Reconstruction MSE (Primary) : {avg_mse_stiff:.2e}")
    print(f"Pressure Reconstruction MSE (Capacity) : {avg_mse_pressure:.2e}")
    print(f"Alpha Reconstruction MSE               : {avg_mse_alpha:.2e}")
    print(f"Parameter Prediction MSE               : {avg_mse_params:.2e}")
    print("="*50)

if __name__ == "__main__":
    evaluate_model_performance()