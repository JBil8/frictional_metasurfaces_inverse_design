import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import mlflow
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.plotting import plot_reconstruction
from utils.early_stopping import EarlyStopping
from utils.interpolation import batched_interp1d
from ml_models.model_mlp import SurfaceInverseModel
from ml_models.loss import CurriculumIntensiveLoss 
from physics.differentiable import AxisymmetricContactLayer

def main():
    cfg = load_config("config.yaml")
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    n_asperities = cfg['physics']['n_asperities']
    
    early_stopping = EarlyStopping(patience=100, verbose=True, delta=1e-5)
    mlflow.set_experiment(cfg['experiment_name'])

    with mlflow.start_run():
        mlflow.log_params(cfg['physics'])
        mlflow.log_params(cfg['training'])

        # ---------------------------------------------------------
        # 1. LOAD AND NORMALIZE DATA (PRESERVING -1.0 PADDING)
        # ---------------------------------------------------------
        print(f"Loading data from {cfg['data']['path']}...")
        data = torch.load(cfg['data']['path'])
        X = data["x"]  # (N, 3, Steps) -> [P_grid, Alpha_padded, S_padded]
        Y = data["y"]  # (N, Params)
        
        global_P_max = data["p_star_max"]
        
        # Calculate maximums ONLY on valid physical data points
        valid_alpha = X[:, 1, :][X[:, 1, :] != -1.0]
        valid_stiff = X[:, 2, :][X[:, 2, :] != -1.0]
        MAX_ALPHA = valid_alpha.max().item()
        MAX_S = valid_stiff.max().item()

        print(f"Norm limits -> P_max: {global_P_max:.4e}, Alpha_max: {MAX_ALPHA:.4e}, S_max: {MAX_S:.4e}")
        mlflow.log_metric("norm_max_pressure", global_P_max)
        mlflow.log_metric("norm_max_alpha", MAX_ALPHA)
        mlflow.log_metric("norm_max_stiff", MAX_S)

        # Scale valid regions; leave padding at exactly -1.0
        X[:, 0, :] = X[:, 0, :] / global_P_max
        X[:, 1, :] = torch.where(X[:, 1, :] != -1.0, X[:, 1, :] / MAX_ALPHA, X[:, 1, :])
        X[:, 2, :] = torch.where(X[:, 2, :] != -1.0, X[:, 2, :] / MAX_S, X[:, 2, :])

        dataset = TensorDataset(X, Y)
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_len, val_len, test_len], generator=generator
        )

        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
        val_plot_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

        model = SurfaceInverseModel(cfg).to(device)
        physics = AxisymmetricContactLayer(cfg=cfg).to(device)

        # Physics simulation requires indentations
        max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        steps = cfg['data']['n_steps']
        indentations = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)
        
        # The true unscaled grid we interpolate against
        p_star_grid = torch.linspace(0, global_P_max, steps).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['training']['epochs'], eta_min=1e-6
        )

        w_stiff = cfg['training']['loss_weights'].get('w_stiff', 1.0)
        w_alpha = cfg['training']['loss_weights'].get('w_pressure', 2.0)
        criterion = CurriculumIntensiveLoss(w_stiff=w_stiff, w_pressure=w_alpha, max_delta=max_d).to(device)

        epochs = cfg['training']['epochs']
        k_start = 1e6
        k_end = 1e8

        print("Starting training...")

        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0
            
            # --- DUAL SCHEDULE ---
            current_k = k_start * (k_end / k_start) ** (epoch / epochs)
            progress = epoch / (epochs * 0.5)
            lambda_param = max(0.0, 1.0 - progress)

            mlflow.log_metric("k_steepness", current_k, step=epoch)
            mlflow.log_metric("lambda_param", lambda_param, step=epoch)

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                # bx is [P_grid, Alpha_padded, S_padded]
                target_alpha = bx[:, 1:2, :]   
                target_stiff = bx[:, 2:3, :]  

                p_n, p_h = model(bx) 
                p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(bx.shape[0], 1)
                
                # 1. Native Physics (Displacement space)
                P_native, alpha_native, S_native = physics(p_h, p_n, p_w, batch_ind, k_steepness=current_k)

                # 2. Interpolate to target P* Grid
                aligned_alpha = batched_interp1d(p_star_grid, P_native, alpha_native, pad_value=-1.0)
                aligned_S = batched_interp1d(p_star_grid, P_native, S_native, pad_value=-1.0)

                # 3. Normalize interpolated predictions (Preserve padding)
                pred_alpha_norm = torch.where(aligned_alpha != -1.0, aligned_alpha / MAX_ALPHA, -1.0)
                pred_S_norm = torch.where(aligned_S != -1.0, aligned_S / MAX_S, -1.0)

                # 4. Rigorous Masked Loss
                total_loss = criterion(
                    pred_stiff=pred_S_norm.unsqueeze(1), 
                    target_stiff=target_stiff,
                    pred_alpha=pred_alpha_norm.unsqueeze(1),
                    target_alpha=target_alpha,
                    pred_params=torch.cat([p_n, p_h], dim=1),
                    target_params=by,
                    lambda_param=lambda_param  
                )
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_accum += total_loss.item()

            avg_train_loss = train_loss_accum / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # --- PHASE 2: VALIDATION ---
            model.eval()
            val_loss_accum = 0.0

            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    
                    val_target_alpha = vx[:, 1:2, :]
                    val_target_stiff = vx[:, 2:3, :]
                    
                    vn, vh = model(vx)
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    batch_ind_val = indentations.repeat(vx.shape[0], 1)
                    
                    # Validation MUST use hard physical cliffs (k_end)
                    v_P, v_alpha, v_S = physics(vh, vn, vw, batch_ind_val, k_steepness=k_end)

                    v_aligned_alpha = batched_interp1d(p_star_grid, v_P, v_alpha, pad_value=-1.0)
                    v_aligned_S = batched_interp1d(p_star_grid, v_P, v_S, pad_value=-1.0)

                    v_alpha_norm = torch.where(v_aligned_alpha != -1.0, v_aligned_alpha / MAX_ALPHA, -1.0)
                    v_S_norm = torch.where(v_aligned_S != -1.0, v_aligned_S / MAX_S, -1.0)

                    val_batch_loss = criterion(
                        pred_stiff=v_S_norm.unsqueeze(1),
                        target_stiff=val_target_stiff,
                        pred_alpha=v_alpha_norm.unsqueeze(1),
                        target_alpha=val_target_alpha,
                        pred_params=torch.cat([vn, vh], dim=1),
                        target_params=vy,
                        lambda_param=0.0  
                    )
                    
                    val_loss_accum += val_batch_loss.item()

            avg_val_loss = val_loss_accum / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            scheduler.step() # Cosine scheduler does not take avg_val_loss
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                mlflow.log_metric("stopped_epoch", epoch)
                break

            # --- PHASE 3: VISUALIZATION ---
            if epoch % 5 == 0:
                with torch.no_grad():
                    vx, vy = next(iter(val_plot_loader))
                    vx, vy = vx.to(device), vy.to(device)

                    vn, vh = model(vx)
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    
                    rec_P, rec_alpha, rec_S = physics(vh, vn, vw, indentations, k_steepness=k_end)
                    al_alpha = batched_interp1d(p_star_grid, rec_P, rec_alpha, pad_value=-1.0)
                    al_S = batched_interp1d(p_star_grid, rec_P, rec_S, pad_value=-1.0)

                    # Extract valid regions for plotting
                    t_a = (vx[0, 1, :] * MAX_ALPHA).cpu().numpy()
                    t_s = (vx[0, 2, :] * MAX_S).cpu().numpy() 
                    p_a = al_alpha[0].cpu().numpy()
                    p_s = al_S[0].cpu().numpy()              
                    
                    p_params = torch.cat([vn, vh], dim=1)[0].cpu().numpy()
                    t_params = vy[0].cpu().numpy()

                    p_grid_np = p_star_grid.cpu().numpy()
                    fig = plot_reconstruction(p_grid_np, t_a, t_s, p_a, p_s, t_params, p_params, epoch)
                    mlflow.log_figure(fig, f"validation_plots/epoch_{epoch}.png")
                    plt.close(fig)

        # --- END OF TRAINING ---
        best_model_name = cfg['model']['name'] 
        early_stopping.save_to_disk(best_model_name)
        
        print("Training complete")

if __name__ == "__main__":
    main()