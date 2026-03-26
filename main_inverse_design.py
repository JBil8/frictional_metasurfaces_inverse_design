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
from ml_models.loss import CurriculumIntensiveLoss 
from physics.differentiable import AxisymmetricContactLayer

def main():
    # 1. Setup
    cfg = load_config("config.yaml")
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    n_asperities = cfg['physics']['n_asperities']
    
    early_stopping = EarlyStopping(patience=25, verbose=True, delta=1e-4)
    mlflow.set_experiment(cfg['experiment_name'])

    with mlflow.start_run():
        mlflow.log_params(cfg['physics'])
        mlflow.log_params(cfg['training'])
        mlflow.log_params(cfg['model'])

        # 2. Load Data
        print(f"Loading data from {cfg['data']['path']}...")
        data = torch.load(cfg['data']['path'])
        X = data["x"]  # (N, 3, Steps) -> [Pressure, Alpha, dP/dAlpha]
        Y = data["y"]  # (N, Params)

        # Normalization (Using the new intensive limits)
        limits = get_theoretical_limits(cfg, device)
        MAX_P = limits['max_pressure']
        MAX_ALPHA = limits['max_alpha']
        MAX_S = limits['max_stiff']
        
        X[:, 0, :] /= MAX_P
        X[:, 1, :] /= MAX_ALPHA
        X[:, 2, :] /= MAX_S  

        mlflow.log_metric("norm_max_pressure", float(MAX_P))
        mlflow.log_metric("norm_max_alpha", float(MAX_ALPHA))
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

        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
        val_plot_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

        model = SurfaceInverseModel(cfg).to(device)
        # CRITICAL FIX: Physics layer now requires the full cfg to calculate domain L
        physics = AxisymmetricContactLayer(cfg=cfg).to(device)

        max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        steps = cfg['data']['n_steps']
        indentations = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=cfg['training']['scheduler']['factor'], 
            patience=cfg['training']['scheduler']['patience']
        )

        # Initialize the Intensive Loss using config weights
        w_stiff = cfg['training']['loss_weights'].get('w_stiff', 1.0)
        w_pressure = cfg['training']['loss_weights'].get('w_pressure', 2.0)
        criterion = CurriculumIntensiveLoss(w_stiff=w_stiff, w_pressure=w_pressure, max_delta=max_d).to(device)

        print("Starting training...")
        epochs = cfg['training']['epochs']

        for epoch in range(epochs):
            model.train()
            train_loss_accum = 0.0

            # CALCULATE DECAY: Fades from 1.0 to 0.0 over the first 50% of epochs
            progress = epoch / (epochs * 0.5)
            lambda_param = max(0.0, 1.0 - progress)
            mlflow.log_metric("lambda_param", lambda_param, step=epoch)

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                target_pressure = bx[:, 0:1, :]   
                target_stiff = bx[:, 2:3, :]  

                # Pass ALL 3 Channels to the network
                p_n, p_h = model(bx) 

                p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(bx.shape[0], 1)
                
                rec_p, rec_alpha, rec_s = physics(p_h, p_n, p_w, batch_ind)

                rec_p = (rec_p / MAX_P).unsqueeze(1)
                rec_s = (rec_s / MAX_S).unsqueeze(1) 

                total_loss = criterion(
                    pred_stiff=rec_s, 
                    target_stiff=target_stiff,
                    pred_pressure=rec_p,
                    target_pressure=target_pressure,
                    pred_params=torch.cat([p_n, p_h], dim=1),
                    target_params=by,
                    lambda_param=lambda_param  # Inject the current decay weight
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
                    
                    val_target_pressure = vx[:, 0:1, :]
                    val_target_stiff = vx[:, 2:3, :]
                    
                    # Passing full 3-channel vx as discussed
                    vn, vh = model(vx)

                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    batch_ind_val = indentations.repeat(vx.shape[0], 1)
                    v_p, v_alpha, v_s = physics(vh, vn, vw, batch_ind_val)

                    v_p = (v_p / MAX_P).unsqueeze(1)
                    v_s = (v_s / MAX_S).unsqueeze(1)

                    val_batch_loss = criterion(
                        pred_stiff=v_s,
                        target_stiff=val_target_stiff,
                        pred_pressure=v_p,
                        target_pressure=val_target_pressure,
                        pred_params=torch.cat([vn, vh], dim=1),
                        target_params=vy,
                        lambda_param=0.0  
                    )
                    
                    val_loss_accum += val_batch_loss.item()

            avg_val_loss = val_loss_accum / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            scheduler.step(avg_val_loss)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            early_stopping(avg_val_loss, model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                mlflow.log_metric("stopped_epoch", epoch)
                break

            # --- PHASE 3: VISUALIZATION ---
            if epoch % 10 == 0:
                with torch.no_grad():
                    vx, vy = next(iter(val_plot_loader))
                    vx, vy = vx.to(device), vy.to(device)

                    val_target_stiff = vx[:, 2:3, :]
                    vn, vh = model(vx)
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    rec_p, rec_alpha, rec_s = physics(vh, vn, vw, indentations)

                    # Extract Target vs Predicted for plotting
                    t_p = (vx[0, 0, :] * MAX_P).cpu().numpy()
                    t_s = (vx[0, 2, :] * MAX_S).cpu().numpy() 
                    p_p = rec_p[0].cpu().numpy()
                    p_s = rec_s[0].cpu().numpy()              
                    
                    p_params = torch.cat([vn, vh], dim=1)[0].cpu().numpy()
                    t_params = vy[0].cpu().numpy()

                    # Passes Pressure and Stiffness to the plotter
                    fig = plot_reconstruction(t_p, t_s, p_p, p_s, t_params, p_params, epoch)
                    mlflow.log_figure(fig, f"validation_plots/epoch_{epoch}.png")
                    plt.close(fig)

        # --- END OF TRAINING ---
        if early_stopping.early_stop:
            print("Early stopping triggered!")
        
        best_model_name = cfg['model']['name'] 
        early_stopping.save_to_disk(best_model_name)
        model.load_state_dict(early_stopping.best_state)
        
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
                
                test_target_stiff = tx[:, 2:3, :]
                pn, ph = model(tx)
                
                pw = torch.ones_like(pn) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(tx.shape[0], 1)
                rp, ra, rs = physics(ph, pn, pw, batch_ind)
                
                rs /= MAX_S
                
                # Test Metrics: Strictly evaluating Stiffness MSE
                l_phys = nn.MSELoss()(rs, tx[:, 2, :])
                test_loss_curve += l_phys.item()
                
                l_param = nn.MSELoss()(pn, ty[:, :n_asperities]) + nn.MSELoss()(ph, ty[:, n_asperities:])
                test_loss_params += l_param.item()

        avg_test_curve_err = test_loss_curve / len(test_loader)
        avg_test_param_err = test_loss_params / len(test_loader)

        print(f"FINAL TEST RESULTS: Stiffness Curve MSE: {avg_test_curve_err:.6f}, Param MSE: {avg_test_param_err:.6f}")
        mlflow.log_metric("test_stiffness_mse", avg_test_curve_err)
        mlflow.log_metric("test_param_mse", avg_test_param_err)

if __name__ == "__main__":
    main()