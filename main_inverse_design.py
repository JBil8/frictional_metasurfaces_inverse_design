import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import mlflow
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.plotting import plot_reconstruction
from ml_models.model_mlp import SurfaceInverseModel
from physics.differentiable import AxisymmetricContactLayer

def main():
    # 1. Setup
    cfg = load_config("config.yaml")
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # Start MLflow Run
    mlflow.set_experiment(cfg['experiment_name'])
    
    with mlflow.start_run():
        # Log all parameters from config
        mlflow.log_params(cfg['physics'])
        mlflow.log_params(cfg['training'])
        mlflow.log_params(cfg['model'])
        
        # 2. Load Data
        print(f"Loading data from {cfg['data']['path']}...")
        data = torch.load(cfg['data']['path'])
        X = data["x"] # (N, 2, Steps)
        Y = data["y"] # (N, Params)
        
        # Normalization (Crucial for stability)
        max_load = X[:, 0, :].max()
        max_area = X[:, 1, :].max()
        max_stiff = X[:, 2, :].max() 
        X[:, 0, :] /= max_load
        X[:, 1, :] /= max_area
        X[:, 2, :] /= max_stiff
        
        # Log normalization factors (needed for inference later!)
        mlflow.log_metric("norm_max_load", float(max_load))
        mlflow.log_metric("norm_max_area", float(max_area))
        mlflow.log_metric("norm_max_stiff", float(max_stiff))

        dataset = TensorDataset(X, Y)
        total_len = len(dataset)

        # Define split sizes (e.g., 80% Train, 10% Val, 10% Test)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        # Perform the random split
        train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

        # Create Loaders
        # Shuffle Train to break correlations
        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
        # Val/Test don't need shuffle, but batch_size=1 helps for detailed analysis
        val_loader = DataLoader(val_ds, batch_size=1) 
        test_loader = DataLoader(test_ds, batch_size=1)

        # Initialize Components
        model = SurfaceInverseModel(cfg).to(device)
        physics = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
        
        # Generate the fixed indentation grid used for all samples
        # (Assuming linear ramp from 0 to max_delta)
        max_d = cfg['physics']['max_delta_ratio'] * cfg['physics']['radius']
        steps = cfg['data']['n_steps']
        indentations = torch.linspace(0, max_d, steps).unsqueeze(0).to(device) # (1, Steps)

        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
        
        # Training Loop
        print("Starting training...")
        epochs = cfg['training']['epochs']
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0  # Reset metric
            
            # Dynamic Reg Decay
            progress = epoch / (epochs * 0.5)
            lambda_reg = max(0.0, 1.0 - progress)
            mlflow.log_metric("lambda_reg", lambda_reg, step=epoch)

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                
                optimizer.zero_grad()  # 1. Clear gradients
                
                # Predict Parameters
                p_n, p_h = model(bx)
                
                # Reconstruct Curve (Physics Forward)
                p_w = torch.ones_like(p_n) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(bx.shape[0], 1)
                
                rec_l, rec_a = physics(p_h, p_n, p_w, batch_ind)
                
                # Normalize reconstruction
                rec_l = rec_l / max_load
                rec_a = rec_a / max_area
                
                # --- LOSS CALCULATION ---
                # 1. Logarithmic Loss (Fixes the magnitude bias)
                loss_log_load = nn.MSELoss()(torch.log1p(rec_l), torch.log1p(bx[:, 0, :]))
                loss_log_area = nn.MSELoss()(torch.log1p(rec_a), torch.log1p(bx[:, 1, :]))
                
                # 2. Slope Loss (Fixes the shape/exponents)
                target_slope = bx[:, 0, 1:] - bx[:, 0, :-1] 
                recon_slope  = rec_l[:, 1:] - rec_l[:, :-1]
                loss_slope = nn.MSELoss()(recon_slope, target_slope)

                # 3. Combine Physics Losses
                loss_phys = (loss_log_load + loss_log_area) * 10.0 + (loss_slope * 100.0)

                # 4. Parameter Regularization
                n_asperities = cfg['physics']['n_asperities']
                loss_param = 5.0 * nn.MSELoss()(p_n, by[:, :n_asperities]) + \
                             1.0 * nn.MSELoss()(p_h, by[:, n_asperities:])

                total_loss = loss_phys + (lambda_reg * loss_param)
                
                # 
                total_loss.backward()  # 2. Calculate gradients
                optimizer.step()       # 3. Update weights
                
                epoch_loss += total_loss.item() # Accumulate for logging

            # Log Avg Train Loss
            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Log Avg Train Loss
            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    # Get one random sample from val set
                    vx, vy = next(iter(val_loader)) 
                    vx = vx.to(device)
                    vy = vy.to(device) # We need the GT params now!
                    
                    vn, vh = model(vx)
                    
                    # Reconstruct
                    vw = torch.ones_like(vn) * 2.0 * cfg['physics']['radius']
                    rec_l, rec_a = physics(vh, vn, vw, indentations)
                    
                    # Un-normalize curves
                    t_l = (vx[0, 0, :] * max_load).cpu().numpy()
                    t_a = (vx[0, 1, :] * max_area).cpu().numpy()
                    p_l = rec_l[0].cpu().numpy()
                    p_a = rec_a[0].cpu().numpy()
                    
                    # Prepare Parameters for plotting
                    # Concatenate predicted n and h to match GT shape
                    p_params = torch.cat([vn, vh], dim=1)[0].cpu().numpy()
                    t_params = vy[0].cpu().numpy()
                    
                    # --- NEW PLOTTING CALL ---
                    fig = plot_reconstruction(t_l, t_a, p_l, p_a, t_params, p_params, epoch)
                    mlflow.log_figure(fig, f"validation_plots/epoch_{epoch}.png")
                    plt.close(fig)
                    
                    print(f"Epoch {epoch}: Logged validation plot.")

        # Save Model artifact
        torch.save(model.state_dict(), "model_final.pth")
        mlflow.log_artifact("model_final.pth")
        print("Training complete. Check MLflow dashboard.")

        print("\nRunning Final Test on unseen data...")
        model.eval()
        test_loss_curve = 0.0
        test_loss_params = 0.0
        
        with torch.no_grad():
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                
                # Predict
                pn, ph = model(tx)
                
                # Reconstruct
                pw = torch.ones_like(pn) * 2.0 * cfg['physics']['radius']
                batch_ind = indentations.repeat(tx.shape[0], 1)
                rl, ra = physics(ph, pn, pw, batch_ind)
                
                # Normalize
                rl /= max_load
                ra /= max_area
                
                # Accumulate Errors
                # Error on Curve (Physics)
                l_phys = nn.MSELoss()(rl, tx[:, 0, :]) + nn.MSELoss()(ra, tx[:, 1, :])
                test_loss_curve += l_phys.item()
                
                
                # Error on Params (MSE)
                l_param = nn.MSELoss()(pn, ty[:, :n_asperities]) + nn.MSELoss()(ph, ty[:, n_asperities:])
                test_loss_params += l_param.item()

        avg_test_curve_err = test_loss_curve / len(test_loader)
        avg_test_param_err = test_loss_params / len(test_loader)
        
        print(f"FINAL TEST RESULTS:")
        print(f"  > Curve Reconstruction MSE: {avg_test_curve_err:.6f}")
        print(f"  > Parameter Prediction MSE: {avg_test_param_err:.6f}")
        
        mlflow.log_metric("test_curve_mse", avg_test_curve_err)
        mlflow.log_metric("test_param_mse", avg_test_param_err)

if __name__ == "__main__":
    main()