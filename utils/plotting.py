import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(t_p, t_s, p_p, p_s, t_params, p_params, epoch):
    """
    Plots the intensive reconstruction during training.
    t_p, p_p: Target and Predicted Nominal Pressure
    t_s, p_s: Target and Predicted Stiffness (dP/dAlpha)
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Epoch {epoch} Reconstruction (Intensive Properties)", fontsize=16)

    # 1. Nominal Pressure vs Indentation
    axs[0].plot(t_p, 'k-', lw=2, label='Target Pressure')
    axs[0].plot(p_p, 'b--', lw=2, label='Predicted Pressure')
    axs[0].set_title("Nominal Pressure Capacity")
    axs[0].set_xlabel("Indentation Step")
    axs[0].set_ylabel("Normalized Pressure (P / P_max)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # 2. Intensive Stiffness vs Indentation
    axs[1].plot(t_s, 'k-', lw=2, label='Target dP/dAlpha')
    axs[1].plot(p_s, 'b--', lw=2, label='Predicted dP/dAlpha')
    axs[1].set_title("Topological Stiffness (Cliffs)")
    axs[1].set_xlabel("Indentation Step")
    axs[1].set_ylabel("Normalized Stiffness")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # 3. Parameters (Topology)
    n_asp = len(t_params) // 2
    t_h = t_params[n_asp:]
    p_h = p_params[n_asp:]
    
    indices = np.arange(n_asp)
    width = 0.35
    
    axs[2].bar(indices - width/2, t_h, width, label='Target Heights', color='k', alpha=0.6)
    axs[2].bar(indices + width/2, p_h, width, label='Predicted Heights', color='b', alpha=0.6)
    axs[2].set_title("Asperity Height Offsets")
    axs[2].set_xlabel("Asperity Index")
    axs[2].set_ylabel("Height [m]")
    axs[2].legend()

    plt.tight_layout()
    return fig