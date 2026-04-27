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
    patience = cfg['training']['patience']
    gamma_min = cfg['physics']['gamma_min']
    gamma_max = cfg['physics']['gamma_max']

    
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-5)
    mlflow.set_experiment(cfg['experiment_name'])

    with mlflow.start_run():
        mlflow.log_params(cfg['physics'])
        mlflow.log_params(cfg['training'])

        # ---------------------------------------------------------
        # LOAD NORMALIZED DATA
        # ---------------------------------------------------------
        print(f"Loading data from {cfg['data']['path']}...")
        data = torch.load(cfg['data']['path'])
        
        X_arrays = data["x_arrays"]    # (N, 2, Steps) -> [Alpha_hat, Stiff_hat]
        X_scalars = data["x_scalars"]  # (N, 2) -> [P_max, Alpha_max]
        Y = data["y"]                  # (N, Params)
        global_P_max = data.get("p_star_max_global", 1.0) 

        print(f"Dataset loaded. Total samples: {len(Y)}")
        mlflow.log_metric("global_p_max", global_P_max)

        dataset = TensorDataset(X_arrays, X_scalars, Y)
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_len, val_len, test_len], generator=generator
        )

        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
        val_plot_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

        # ---------------------------------------------------------
        # INITIALIZE ENGINE & MODEL
        # ---------------------------------------------------------
        model = SurfaceInverseModel(cfg).to(device)
        physics = AxisymmetricContactLayer(cfg=cfg).to(device)

        max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        steps = cfg['data']['n_steps']
        indentations = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)
        
        # The universal normalized grid for all interpolation [0.0 to 1.0]
        p_hat_grid = torch.linspace(0, 1.0, steps).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['training']['epochs'], eta_min=cfg['training']['scheduler']['eta_min'])

        # Initialize the simplified loss function
        criterion = CurriculumIntensiveLoss(
            w_shape=cfg['training']['loss_weights'].get('w_shape', 10.0),
            w_grad=cfg['training']['loss_weights'].get('w_grad', 1.0),
            w_mag=cfg['training']['loss_weights'].get('w_mag', 1.0),
            max_delta=max_d,
            gamma_max=gamma_max,
        ).to(device)

        epochs = cfg['training']['epochs']
        k_start = 1e3
        k_end = 1e5

        print("Starting training...")

        for epoch in range(epochs):
            # --- PHASE 1: TRAINING ---
            model.train()
            train_loss_accum = 0.0
            
            # Dual Schedule
            scale_lambda = 0.1
            progress_k = 4 * epoch / epochs 
            progress_lambda = 8 * epoch / epochs
            current_k = min(k_start * (k_end / k_start) ** progress_k, k_end)
            lambda_param = scale_lambda * max(0.0, 1.0 - progress_lambda)
            
            mlflow.log_metric("k_steepness", current_k, step=epoch)
            mlflow.log_metric("lambda_param", lambda_param, step=epoch)

            for bx_arr, bx_scal, by in train_loader:
                bx_arr = bx_arr.to(device)
                bx_scal = bx_scal.to(device)
                by = by.to(device)
                
                optimizer.zero_grad()

                # Extract shape targets directly (no masks needed)
                target_alpha_hat = bx_arr[:, 0, :]   
                target_stiff_hat = bx_arr[:, 1, :]  

                # Log transform scalars for numerical stability during forward pass
                bx_scal_log = torch.log10(bx_scal + 1e-12)

                # Forward Neural Network
                p_n, p_h = model(bx_arr, bx_scal_log) 
                p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(bx_arr.shape[0], 1)
                
                # Forward Sneddon Physics (Native space)
                P_nat, alpha_nat, S_nat = physics(p_h, p_n, p_w, batch_ind, k_steepness=current_k)

                # Extract predicted scalars
                pred_p_max = torch.clamp(P_nat[:, -1], min=1e-12)
                pred_a_max = torch.clamp(alpha_nat[:, -1], min=1e-12)
                pred_scalars = torch.stack([pred_p_max, pred_a_max], dim=1)

                # Normalize predicted arrays locally
                P_hat_pred = P_nat / pred_p_max.unsqueeze(1)
                alpha_hat_pred = alpha_nat / pred_a_max.unsqueeze(1)
                S_hat_pred = S_nat * (pred_a_max / pred_p_max).unsqueeze(1)

                # Interpolate predictions onto the strict 0-to-1 grid
                pred_alpha_interp = batched_interp1d(p_hat_grid, P_hat_pred, alpha_hat_pred, pad_value=1.0)
                pred_S_interp = batched_interp1d(p_hat_grid, P_hat_pred, S_hat_pred, pad_value=0.0)

                # Calculate rigorous split loss
                total_loss = criterion(
                    pred_alpha_hat=pred_alpha_interp, 
                    target_alpha_hat=target_alpha_hat,
                    pred_stiff_hat=pred_S_interp,
                    target_stiff_hat=target_stiff_hat,
                    pred_scalars=pred_scalars,
                    target_scalars=bx_scal,
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
                for vx_arr, vx_scal, vy in val_loader:
                    vx_arr, vx_scal, vy = vx_arr.to(device), vx_scal.to(device), vy.to(device)
                    
                    val_targ_alpha = vx_arr[:, 0, :]
                    val_targ_stiff = vx_arr[:, 1, :]
                    
                    vx_scal_log = torch.log10(vx_scal + 1e-12)
                    vn, vh = model(vx_arr, vx_scal_log)
                    
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    batch_ind_val = indentations.repeat(vx_arr.shape[0], 1)
                    
                    # Validation MUST use hard physical cliffs (k_end)
                    v_P, v_alpha, v_S = physics(vh, vn, vw, batch_ind_val, k_steepness=k_end)

                    v_p_max = torch.clamp(v_P[:, -1], min=1e-12)
                    v_a_max = torch.clamp(v_alpha[:, -1], min=1e-12)
                    v_scalars = torch.stack([v_p_max, v_a_max], dim=1)

                    v_P_hat = v_P / v_p_max.unsqueeze(1)
                    v_alpha_hat = v_alpha / v_a_max.unsqueeze(1)
                    v_S_hat = v_S * (v_a_max / v_p_max).unsqueeze(1)

                    v_alpha_interp = batched_interp1d(p_hat_grid, v_P_hat, v_alpha_hat, pad_value=1.0)
                    v_S_interp = batched_interp1d(p_hat_grid, v_P_hat, v_S_hat, pad_value=0.0)

                    val_batch_loss = criterion(
                        pred_alpha_hat=v_alpha_interp,
                        target_alpha_hat=val_targ_alpha,
                        pred_stiff_hat=v_S_interp,
                        target_stiff_hat=val_targ_stiff,
                        pred_scalars=v_scalars,
                        target_scalars=vx_scal,
                        pred_params=torch.cat([vn, vh], dim=1),
                        target_params=vy,
                        lambda_param=0.0  
                    )
                    val_loss_accum += val_batch_loss.item()

            avg_val_loss = val_loss_accum / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            scheduler.step()
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                mlflow.log_metric("stopped_epoch", epoch)
                break

            # --- PHASE 3: VISUALIZATION ---
            if epoch % 5 == 0:
                with torch.no_grad():
                    vx_arr, vx_scal, vy = next(iter(val_plot_loader))
                    vx_arr, vx_scal, vy = vx_arr.to(device), vx_scal.to(device), vy.to(device)

                    vx_scal_log = torch.log10(vx_scal + 1e-12)
                    vn, vh = model(vx_arr, vx_scal_log)
                    
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    
                    # 1. Raw native prediction from the physics engine (Absolute space)
                    rec_P, rec_alpha, rec_S = physics(vh, vn, vw, indentations, k_steepness=k_end)
                    
                    # 2. Extract arrays and scalars to CPU
                    t_a_hat = vx_arr[0, 0, :].cpu().numpy()
                    t_s_hat = vx_arr[0, 1, :].cpu().numpy() 
                    
                    t_p_max = vx_scal[0, 0].cpu().numpy()
                    t_a_max = vx_scal[0, 1].cpu().numpy()
                    
                    p_grid_np = p_hat_grid.cpu().numpy() # [0.0 to 1.0]

                    # 3. DENORMALIZE THE TARGET back to physical reality
                    t_p_abs = p_grid_np * t_p_max
                    t_a_abs = t_a_hat * t_a_max
                    t_s_abs = t_s_hat * (t_p_max / t_a_max)

                    # 4. Grab the absolute predicted arrays directly (no interpolation needed)
                    # Because we will plot y vs x, it doesn't matter that the x-axes (P) 
                    # differ slightly between target and prediction.
                    p_p_abs = rec_P[0].cpu().numpy()
                    p_a_abs = rec_alpha[0].cpu().numpy()
                    p_s_abs = rec_S[0].cpu().numpy()
                    
                    # Extract predicted maxes for the plot title
                    p_p_max = p_p_abs[-1]
                    p_a_max = p_a_abs[-1]

                    p_params = torch.cat([vn, vh], dim=1)[0].cpu().numpy()
                    t_params = vy[0].cpu().numpy()

                    # 5. Update your plot_reconstruction to accept the separate X-axes and scalars
                    fig = plot_reconstruction(
                        t_p_abs, t_a_abs, t_s_abs,  # Target X, Y1, Y2
                        p_p_abs, p_a_abs, p_s_abs,  # Pred X, Y1, Y2
                        t_params, p_params, epoch,
                        t_p_max, t_a_max,           # Pass scalars for title/text
                        p_p_max, p_a_max,
                        gamma_min, gamma_max        # Pass config bounds for reference lines
                    )
                    
                    mlflow.log_figure(fig, f"validation_plots/epoch_{epoch}.png")
                    plt.close(fig)

        # --- END OF TRAINING ---
        best_model_name = cfg['model']['name'] 
        early_stopping.save_to_disk(best_model_name)
        
        print("Training complete")

if __name__ == "__main__":
    main()
