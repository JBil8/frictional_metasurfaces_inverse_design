import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config
from utils.interpolation import batched_interp1d
from ml_models.model_mlp import SurfaceInverseModel
from ml_models.loss import CurriculumIntensiveLoss
from physics.differentiable import AxisymmetricContactLayer

def evaluate_split(loader, model, physics, criterion, p_hat_grid, cfg, device):
    """
    Evaluates a specific data loader using strict physical parameters.
    """
    total_loss_accum = 0.0
    total_samples = 0
    
    # Strict evaluation parameters
    k_eval = 1e5           # Fully annealed macroscopic cliffs
    lambda_eval = 0.0      # pure physics loss

    for x_arr, x_scal, y_targ in loader:
        x_arr = x_arr.to(device)
        x_scal = x_scal.to(device)
        y_targ = y_targ.to(device)
        
        batch_size = x_arr.shape[0]
        
        # 1. Extract Targets
        target_alpha_hat = x_arr[:, 0, :]   
        target_stiff_hat = x_arr[:, 1, :]  
        
        # 2. Forward Neural Network
        x_scal_log = torch.log10(x_scal + 1e-12)
        p_n, p_h = model(x_arr, x_scal_log) 
        
        # 3. Reconstruct Physics
        p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
        max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        indentations = torch.linspace(0, max_d, cfg['data']['n_steps']).repeat(batch_size, 1).to(device)
        
        rec_P, rec_alpha, rec_S = physics(p_h, p_n, p_w, indentations, k_steepness=k_eval)
        
        # 4. Extract Predicted Scalars
        pred_p_max = torch.clamp(rec_P[:, -1], min=1e-12)
        pred_a_max = torch.clamp(rec_alpha[:, -1], min=1e-12)
        pred_scalars = torch.stack([pred_p_max, pred_a_max], dim=1)

        # 5. Normalize Predicted Arrays Locally
        P_hat_pred = rec_P / pred_p_max.unsqueeze(1)
        alpha_hat_pred = rec_alpha / pred_a_max.unsqueeze(1)
        S_hat_pred = rec_S * (pred_a_max / pred_p_max).unsqueeze(1)

        # 6. Interpolate predictions onto the strict 0-to-1 grid
        pred_alpha_interp = batched_interp1d(p_hat_grid, P_hat_pred, alpha_hat_pred, pad_value=1.0)
        pred_S_interp = batched_interp1d(p_hat_grid, P_hat_pred, S_hat_pred, pad_value=0.0)

        # 7. Calculate Composite Physics Loss
        loss = criterion(
            pred_alpha_hat=pred_alpha_interp, 
            target_alpha_hat=target_alpha_hat,
            pred_stiff_hat=pred_S_interp,
            target_stiff_hat=target_stiff_hat,
            pred_scalars=pred_scalars,
            target_scalars=x_scal,
            pred_params=torch.cat([p_n, p_h], dim=1),
            target_params=y_targ,
            lambda_param=lambda_eval  
        )
        
        total_loss_accum += loss.item() * batch_size
        total_samples += batch_size

    return total_loss_accum / total_samples


def main():
    cfg = load_config("config.yaml") if os.path.exists("config.yaml") else load_config("../config.yaml")
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device.type.upper()}")

    # ---------------------------------------------------------
    # 1. LOAD AND SPLIT DATA (Exact same seed as training)
    # ---------------------------------------------------------
    print(f"Loading data from {cfg['data']['path']}...")
    data = torch.load(cfg['data']['path'], map_location=device)
    
    dataset = TensorDataset(data["x_arrays"], data["x_scalars"], data["y"])
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    batch_size = 2048
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ---------------------------------------------------------
    # 2. LOAD MODEL IN EVAL MODE (Disables Dropout)
    # ---------------------------------------------------------
    model = SurfaceInverseModel(cfg).to(device)
    
    model_path = cfg['model']['name']
    if not os.path.exists(model_path):
        model_path = os.path.join("..", model_path)
        
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    # ---------------------------------------------------------
    # 3. INITIALIZE PHYSICS & CRITERION
    # ---------------------------------------------------------
    physics = AxisymmetricContactLayer(cfg=cfg).to(device)
    steps = cfg['data']['n_steps']
    p_hat_grid = torch.linspace(0, 1.0, steps).to(device)
    
    # Same loss weights as training config
    criterion = CurriculumIntensiveLoss(
        w_shape=cfg['training']['loss_weights'].get('w_shape', 10.0),
        w_grad=cfg['training']['loss_weights'].get('w_grad', 1.0),
        w_mag=cfg['training']['loss_weights'].get('w_mag', 10.0),
        max_delta=cfg['physics']['max_delta_ratio'] * cfg['physics']['radius'],
        gamma_max=cfg['physics']['gamma_max'],
        gamma_min=cfg['physics']['gamma_min']
    ).to(device)

    # ---------------------------------------------------------
    # 4. EVALUATE
    # ---------------------------------------------------------
    print("\nEvaluating...")
    with torch.no_grad():
        test_loss = evaluate_split(test_loader, model, physics, criterion, p_hat_grid, cfg, device)
        val_loss = evaluate_split(val_loader, model, physics, criterion, p_hat_grid, cfg, device)
        train_loss = evaluate_split(train_loader, model, physics, criterion, p_hat_grid, cfg, device)

    print("="*50)
    print(f"FINAL MODEL PERFORMANCE (Curriculum Loss | lambda=0, k=1e5)")
    print("="*50)
    print(f"Training Loss   : {train_loss:.6f}  ({train_len} samples)")
    print(f"Validation Loss : {val_loss:.6f}  ({val_len} samples)")
    print(f"Test Loss       : {test_loss:.6f}  ({test_len} samples)")
    print("="*50)

if __name__ == "__main__":
    main()