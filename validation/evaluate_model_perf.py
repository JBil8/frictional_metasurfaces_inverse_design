from physics.differentiable import AxisymmetricContactLayer
from ml_models.model_mlp import SurfaceInverseModel
from utils.config import load_config
from utils.normalization import get_theoretical_limits
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_model_performance():
    cfg = load_config("config.yaml")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Running evaluation on: {device_str.upper()}")

    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'], map_location=device)

    X = data["x"]  
    Y = data["y"]  

    dataset = TensorDataset(X, Y)
    total_len = len(dataset)

    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(42)

    _, _, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator
    )

    print(f"Test Set Size: {len(test_ds)} samples")
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)

    limits = get_theoretical_limits(cfg, device)
    MAX_L = limits['max_load']
    MAX_A = limits['max_area']
    MAX_S = limits['max_stiff']

    model = SurfaceInverseModel(cfg).to(device)

    print("Loading model weights...")
    model_name = cfg['model']['name']
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    physics = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
    indentations = torch.linspace(0, max_d, cfg['data']['n_steps']).unsqueeze(0).to(device)

    mse_load = 0.0
    mse_area = 0.0
    mse_stiff = 0.0
    mse_params = 0.0
    total_samples = 0

    criterion = nn.MSELoss(reduction='sum')

    print(f"Starting evaluation loop...")

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            batch_size = bx.size(0)

            bx_norm = bx.clone()
            bx_norm[:, 0, :] /= MAX_L
            bx_norm[:, 1, :] /= MAX_A
            bx_norm[:, 2, :] /= MAX_S

            # CRITICAL FIX: Only pass the Normalized Stiffness Channel (Channel 2)
            p_n, p_h = model(bx_norm[:, 2:3, :])

            p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
            batch_ind = indentations.repeat(batch_size, 1)
            
            # CRITICAL FIX: Unpack 3 variables
            rec_l, rec_a, rec_s = physics(p_h, p_n, p_w, batch_ind)

            rec_l /= MAX_L
            rec_a /= MAX_A
            rec_s /= MAX_S

            loss_l = criterion(rec_l, bx_norm[:, 0, :])
            loss_a = criterion(rec_a, bx_norm[:, 1, :])
            loss_s = criterion(rec_s, bx_norm[:, 2, :]) # The true objective!

            pred_params = torch.cat([p_n, p_h], dim=1)
            loss_p = criterion(pred_params, by)

            mse_load += loss_l.item()
            mse_area += loss_a.item()
            mse_stiff += loss_s.item()
            mse_params += loss_p.item()
            total_samples += batch_size

    n_steps = cfg['data']['n_steps']
    n_params = Y.shape[1]

    avg_mse_load = mse_load / (total_samples * n_steps)
    avg_mse_area = mse_area / (total_samples * n_steps)
    avg_mse_stiff = mse_stiff / (total_samples * n_steps)
    avg_mse_params = mse_params / (total_samples * n_params)

    print("="*40)
    print(f"FINAL TEST RESULTS (Normalized Units)")
    print("="*40)
    print(f"Stiffness Reconstruction MSE (Primary): {avg_mse_stiff:.2e}")
    print(f"Load Reconstruction MSE    (Extensive): {avg_mse_load:.2e}")
    print(f"Area Reconstruction MSE    (Extensive): {avg_mse_area:.2e}")
    print(f"Parameter Prediction MSE              : {avg_mse_params:.2e}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model_performance()