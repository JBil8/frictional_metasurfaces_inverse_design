import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
    # FIX: map_location ensure it loads to CPU if GPU is missing
    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'], map_location=device)
    
    X = data["x"] # (N, 3, Steps)
    Y = data["y"] # (N, Params)
    
    # Normalize (Apply same logic as training)
    max_load = X[:, 0, :].max()
    max_area = X[:, 1, :].max()
    max_stiff = X[:, 2, :].max()
    
    X[:, 0, :] /= max_load
    X[:, 1, :] /= max_area
    X[:, 2, :] /= max_stiff
    
    # Create Test Set (Last 10%)
    total_len = len(X)
    test_start_idx = int(0.9 * total_len)
    
    test_x = X[test_start_idx:]
    test_y = Y[test_start_idx:]
    
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=100)
    
    # Load Model
    model = SurfaceInverseModel(cfg).to(device)
    
    # FIX: map_location here as well for model weights
    print("Loading model weights...")
    state_dict = torch.load("model_final.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Physics Layer
    physics = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
    indentations = torch.linspace(0, max_d, cfg['data']['n_steps']).unsqueeze(0).to(device)

    # 3. Accumulate Errors
    mse_load = 0.0
    mse_area = 0.0
    mse_params = 0.0
    total_samples = 0
    
    criterion = nn.MSELoss(reduction='sum')
    
    print(f"Evaluating on {len(test_x)} test samples...")
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            batch_size = bx.size(0)
            
            # Predict
            p_n, p_h = model(bx)
            
            # Reconstruct Physics
            p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
            batch_ind = indentations.repeat(batch_size, 1)
            rec_l, rec_a = physics(p_h, p_n, p_w, batch_ind)
            
            # Normalize Reconstruction
            rec_l /= max_load
            rec_a /= max_area
            
            # Calculate Errors
            loss_l = criterion(rec_l, bx[:, 0, :])
            loss_a = criterion(rec_a, bx[:, 1, :])
            
            # Parameter Error
            pred_params = torch.cat([p_n, p_h], dim=1)
            loss_p = criterion(pred_params, by)
            
            mse_load += loss_l.item()
            mse_area += loss_a.item()
            mse_params += loss_p.item()
            total_samples += batch_size

    # 4. Final Calculation
    n_steps = cfg['data']['n_steps']
    n_params = test_y.shape[1]
    
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