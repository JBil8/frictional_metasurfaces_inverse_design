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
from ml_models.loss import StiffnessLoss
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
        X = data["x"]  # (N, 3, Steps) -> [Load, Area, dF/dA]
        Y = data["y"]  # (N, Params)

        # Normalization
        limits = get_theoretical_limits(cfg, device)
        MAX_L = limits['max_load']
        MAX_A = limits['max_area']
        MAX_S = limits['max_stiff']
        
        X[:, 0, :] /= MAX_L
        X[:, 1, :] /= MAX_A
        X[:, 2, :] /= MAX_S  # Normalize Stiffness

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

        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
        val_plot_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

        model = SurfaceInverseModel(cfg).to(device)
        physics = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)

        max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        steps = cfg['data']['n_steps']
        indentations = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=cfg['training']['scheduler']['factor'], 
            patience=cfg['training']['scheduler']['patience']
        )

        criterion = StiffnessLoss(w_stiff=1.0, w_grad=0.5).to(device)

        print("Starting training...")
        epochs = cfg['training']['epochs']

        for epoch in range(epochs):
            # --- PHASE 1: TRAINING ---
            model.train()
            train_loss_accum = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                # Isolate targets
                target_load = bx[:, 0:1, :]   # Extract Load
                target_stiff = bx[:, 2:3, :]  # Extract Stiffness

                p_n, p_h = model(target_stiff) # Input is STILL only Stiffness!

                # Reconstruct Physics
                p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(bx.shape[0], 1)
                
                rec_l, rec_a, rec_s = physics(p_h, p_n, p_w, batch_ind)

                # Normalize Predictions
                rec_l = (rec_l / MAX_L).unsqueeze(1)
                rec_s = (rec_s / MAX_S).unsqueeze(1) 

                total_loss = criterion(
                    pred_curve=rec_s, 
                    target_curve=target_stiff
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
                    
                    val_target_load = vx[:, 0:1, :]
                    val_target_stiff = vx[:, 2:3, :]
                    
                    vn, vh = model(val_target_stiff)

                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    batch_ind_val = indentations.repeat(vx.shape[0], 1)
                    v_l, v_a, v_s = physics(vh, vn, vw, batch_ind_val)

                    v_l = (v_l / MAX_L).unsqueeze(1)
                    v_s = (v_s / MAX_S).unsqueeze(1)

                    val_batch_loss = criterion(
                        pred_curve=v_s,
                        target_curve=val_target_stiff
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
                    vn, vh = model(val_target_stiff)
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    rec_l, rec_a, rec_s = physics(vh, vn, vw, indentations)

                    # Extract Target vs Predicted for plotting
                    t_l = (vx[0, 0, :] * MAX_L).cpu().numpy()
                    t_s = (vx[0, 2, :] * MAX_S).cpu().numpy() # True Stiffness
                    p_l = rec_l[0].cpu().numpy()
                    p_s = rec_s[0].cpu().numpy()              # Pred Stiffness
                    
                    p_params = torch.cat([vn, vh], dim=1)[0].cpu().numpy()
                    t_params = vy[0].cpu().numpy()

                    # Passing t_s and p_s instead of Area for the plot
                    fig = plot_reconstruction(t_l, t_s, p_l, p_s, t_params, p_params, epoch)
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
                pn, ph = model(test_target_stiff)
                
                pw = torch.ones_like(pn) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(tx.shape[0], 1)
                rl, ra, rs = physics(ph, pn, pw, batch_ind)
                
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