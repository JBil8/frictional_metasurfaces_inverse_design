import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import mlflow
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.plotting import plot_reconstruction
from utils.normalization import get_theoretical_limits
from utils.early_stopping import EarlyStopping
from ml_models.model_mlp import SurfaceInverseModel
from ml_models.loss import HybridLoss
from physics.differentiable import AxisymmetricContactLayer


# ... imports ...
from ml_models.loss import HybridLoss  # <--- Make sure this is imported

def main():
    # 1. Setup
    cfg = load_config("config.yaml")
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    n_asperities = cfg['physics']['n_asperities']
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=1e-4)
    
    # Start MLflow Run
    mlflow.set_experiment(cfg['experiment_name'])

    with mlflow.start_run():
        # Log all parameters
        mlflow.log_params(cfg['physics'])
        mlflow.log_params(cfg['training'])
        mlflow.log_params(cfg['model'])

        # 2. Load Data
        print(f"Loading data from {cfg['data']['path']}...")
        data = torch.load(cfg['data']['path'])
        X = data["x"]  # (N, 2, Steps)
        Y = data["y"]  # (N, Params)

        # Normalization
        limits = get_theoretical_limits(cfg, device)
        MAX_L = limits['max_load']
        MAX_A = limits['max_area']
        MAX_S = limits['max_stiff']
        X[:, 0, :] /= MAX_L
        X[:, 1, :] /= MAX_A
        X[:, 2, :] /= MAX_S

        # Log normalization factors
        mlflow.log_metric("norm_max_load", float(MAX_L))
        mlflow.log_metric("norm_max_area", float(MAX_A))
        mlflow.log_metric("norm_max_stiff", float(MAX_S))

        dataset = TensorDataset(X, Y)
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_len, val_len, test_len], generator=generator
        )

        # Loaders
        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
        val_plot_loader = DataLoader(val_ds, batch_size=1, shuffle=True) # For plotting only

        # Components
        model = SurfaceInverseModel(cfg).to(device)
        physics = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)

        # Fixed Indentation Grid
        max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        steps = cfg['data']['n_steps']
        indentations = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=cfg['training']['scheduler']['factor'], 
            patience=cfg['training']['scheduler']['patience']
        )

        # --- NEW: INITIALIZE HYBRID LOSS ---
        # Weights can be tuned via Config or Optuna later
        criterion = HybridLoss(w_log=10.0, w_lin=20.0, w_slope=5.0, w_param=1.0).to(device)

        # Training Loop
        print("Starting training...")
        epochs = cfg['training']['epochs']

        for epoch in range(epochs):
            # --- PHASE 1: TRAINING ---
            model.train()
            train_loss_accum = 0.0

            # Dynamic Reg Decay (Optional, affects parameter loss weight if you want)
            progress = epoch / (epochs * 0.5)
            lambda_reg = max(0.0, 1.0 - progress)
            mlflow.log_metric("lambda_reg", lambda_reg, step=epoch)

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                p_n, p_h = model(bx)

                # Reconstruct
                p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(bx.shape[0], 1)
                rec_l, rec_a = physics(p_h, p_n, p_w, batch_ind)

                # Normalize
                rec_l = rec_l / MAX_L
                rec_a = rec_a / MAX_A
                
                rec_curve = torch.stack([rec_l, rec_a], dim=1) # (B, 2, Steps)

                target_curve_sliced = bx[:, :2, :]

                total_loss = criterion(
                    pred_curve=rec_curve, 
                    target_curve=target_curve_sliced, # <--- Use the sliced target
                    pred_params=torch.cat([p_n, p_h], dim=1), 
                    target_params=by
                )
                
                total_loss.backward()
                
                # Clip Gradients (Safety)
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

                    vn, vh = model(vx)

                    # Reconstruct
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    batch_ind_val = indentations.repeat(vx.shape[0], 1)
                    v_l, v_a = physics(vh, vn, vw, batch_ind_val)

                    v_l /= MAX_L
                    v_a /= MAX_A
                    
                    v_curve = torch.stack([v_l, v_a], dim=1)

                    val_target_sliced = vx[:, :2, :]

                    val_batch_loss = criterion(
                        pred_curve=v_curve, 
                        target_curve=val_target_sliced, # <--- Use the sliced target
                        pred_params=torch.cat([vn, vh], dim=1), 
                        target_params=vy
                    )
                    
                    val_loss_accum += val_batch_loss.item()

            avg_val_loss = val_loss_accum / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            # Scheduler & Early Stopping
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                mlflow.log_metric("stopped_epoch", epoch)
                break

            # --- PHASE 3: VISUALIZATION (Use plot loader) ---
            if epoch % 10 == 0:
                with torch.no_grad():
                    # Use the specific plotting loader (Batch=1)
                    vx, vy = next(iter(val_plot_loader))
                    vx, vy = vx.to(device), vy.to(device)

                    vn, vh = model(vx)
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    rec_l, rec_a = physics(vh, vn, vw, indentations)

                    # Un-normalize & Plot
                    t_l = (vx[0, 0, :] * MAX_L).cpu().numpy()
                    t_a = (vx[0, 1, :] * MAX_A).cpu().numpy()
                    p_l = rec_l[0].cpu().numpy()
                    p_a = rec_a[0].cpu().numpy()
                    
                    p_params = torch.cat([vn, vh], dim=1)[0].cpu().numpy()
                    t_params = vy[0].cpu().numpy()

                    fig = plot_reconstruction(t_l, t_a, p_l, p_a, t_params, p_params, epoch)
                    mlflow.log_figure(fig, f"validation_plots/epoch_{epoch}.png")
                    plt.close(fig)
                    print(f"Epoch {epoch}: Logged validation plot.")

        # --- END OF TRAINING ---
        if early_stopping.early_stop:
            print("Early stopping triggered!")
        
        # Save Best Model
        best_model_name = f"model_best_{cfg['data']['n_samples']}.pth"
        early_stopping.save_to_disk(best_model_name)
        model.load_state_dict(early_stopping.best_state) # Load best weights for test
        
        mlflow.log_artifact(best_model_name)
        print("Training complete")

        # --- FINAL TEST ---
        print("\nRunning Final Test on unseen data...")
        model.eval()
        test_loss_curve = 0.0
        test_loss_params = 0.0

        with torch.no_grad():
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                pn, ph = model(tx)
                
                pw = torch.ones_like(pn) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(tx.shape[0], 1)
                rl, ra = physics(ph, pn, pw, batch_ind)
                
                rl /= MAX_L
                ra /= MAX_A
                
                # Test Metrics (Can stay as pure MSE for reporting)
                l_phys = nn.MSELoss()(rl, tx[:, 0, :]) + nn.MSELoss()(ra, tx[:, 1, :])
                test_loss_curve += l_phys.item()
                
                l_param = nn.MSELoss()(pn, ty[:, :n_asperities]) + nn.MSELoss()(ph, ty[:, n_asperities:])
                test_loss_params += l_param.item()

        avg_test_curve_err = test_loss_curve / len(test_loader)
        avg_test_param_err = test_loss_params / len(test_loader)

        print(f"FINAL TEST RESULTS: Curve MSE: {avg_test_curve_err:.6f}, Param MSE: {avg_test_param_err:.6f}")
        mlflow.log_metric("test_curve_mse", avg_test_curve_err)
        mlflow.log_metric("test_param_mse", avg_test_param_err)

if __name__ == "__main__":
    main()
