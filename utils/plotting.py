import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(target_load, target_stiff, pred_load, pred_stiff, 
                        target_params, pred_params, epoch):
    """
    Creates a 2x2 figure comparing Curves AND Parameters.
    
    Args:
        target_load, target_stiff: (N_steps,) arrays
        pred_load, pred_stiff:     (N_steps,) arrays
        target_params: (2*N,) array [N exponents, N offsets]
        pred_params:   (2*N,) array
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Row 1: The Physics (Curves) ---
    
    # 1. Load vs Step
    axs[0, 0].plot(target_load, label='Target (GT)', color='black', linewidth=2)
    axs[0, 0].plot(pred_load, label='Pred', color='red', linestyle='--')
    axs[0, 0].set_title(f"Load vs Indentation Step (Epoch {epoch})")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Stiffness (dF/dA) vs Step
    axs[0, 1].plot(target_stiff, label='Target (GT)', color='black', linewidth=2)
    axs[0, 1].plot(pred_stiff, label='Pred', color='blue', linestyle='--')
    axs[0, 1].set_title("Marginal Friction (dF/dA) vs Indentation Step")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # --- Row 2: The Design (Parameters) ---
    
    # DYNAMIC: Calculate n_asp safely based on the passed array length
    n_asp = len(target_params) // 2
    
    gt_n = target_params[:n_asp]
    pred_n = pred_params[:n_asp]
    
    gt_h = target_params[n_asp:]
    pred_h = pred_params[n_asp:]
    
    x_indices = np.arange(n_asp)
    
    # 3. Exponents Comparison
    width = 0.35
    axs[1, 0].bar(x_indices - width/2, gt_n, width, label='GT Exponents', color='gray')
    axs[1, 0].bar(x_indices + width/2, pred_n, width, label='Pred Exponents', color='orange')
    axs[1, 0].set_title("Shape Exponents (n) per Asperity")
    axs[1, 0].set_xlabel("Asperity Index")
    axs[1, 0].legend()
    
    # 4. Height Offsets Comparison
    axs[1, 1].bar(x_indices - width/2, gt_h, width, label='GT Offsets', color='gray')
    axs[1, 1].bar(x_indices + width/2, pred_h, width, label='Pred Offsets', color='green')
    axs[1, 1].set_title("Height Offsets (h) per Asperity")
    axs[1, 1].set_xlabel("Asperity Index")
    axs[1, 1].legend()
    
    plt.tight_layout()
    return fig