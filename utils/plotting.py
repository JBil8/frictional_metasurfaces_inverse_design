import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(p_grid, t_alpha, t_s, p_alpha, p_s, t_params, p_params, epoch=None):
    """
    Publication-grade 2x2 plotting routine for intensive property reconstruction.
    p_grid: The global P* grid (x-axis)
    t_alpha, p_alpha: Target and Predicted Contact Fraction
    t_s, p_s: Target and Predicted Stiffness (dP/dAlpha)
    t_params: Ground truth [exponents, heights]
    p_params: Predicted [exponents, heights]
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    title = f"Epoch {epoch} Reconstruction" if epoch is not None else "Zero-Shot Network Prediction"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=11)

    # --- Mask out the -1.0 padding for clean physical plotting ---
    t_mask = t_alpha != -1.0
    p_mask = p_alpha != -1.0

    # --- 1. Contact Fraction vs Pressure ---
    axs[0, 0].plot(p_grid[t_mask], t_alpha[t_mask], 'k-', lw=3, label='Target')
    axs[0, 0].plot(p_grid[p_mask], p_alpha[p_mask], 'b--', lw=2.5, label='Predicted')
    axs[0, 0].set_title("Contact Area vs Load", fontsize=14)
    axs[0, 0].set_xlabel("Dimensionless Pressure P*", fontsize=12)
    axs[0, 0].set_ylabel("Contact Fraction α", fontsize=12)
    axs[0, 0].legend(fontsize=11)

    # --- 2. Intensive Stiffness vs Pressure ---
    axs[0, 1].plot(p_grid[t_mask], t_s[t_mask], 'k-', lw=3, label='Target')
    axs[0, 1].plot(p_grid[p_mask], p_s[p_mask], 'b--', lw=2.5, label='Predicted')
    axs[0, 1].set_title("Topological Stiffness (dP/dα)", fontsize=14)
    axs[0, 1].set_xlabel("Dimensionless Pressure P*", fontsize=12)
    axs[0, 1].set_ylabel("Stiffness S*", fontsize=12)
    axs[0, 1].legend(fontsize=11)

    # --- Parameter Extraction & Sorting ---
    n_asp = len(t_params) // 2
    t_n, t_h = t_params[:n_asp], t_params[n_asp:]
    p_n, p_h = p_params[:n_asp], p_params[n_asp:]
    
    t_sort_idx = np.argsort(t_h)
    p_sort_idx = np.argsort(p_h)

    t_h_sorted = t_h[t_sort_idx]
    p_h_sorted = p_h[p_sort_idx]
    t_n_sorted = t_n[t_sort_idx]
    p_n_sorted = p_n[p_sort_idx]

    indices = np.arange(n_asp)
    width = 0.35

    # --- 3. Asperity Height Offsets ---
    axs[1, 0].bar(indices - width/2, t_h_sorted, width, label='Target', color='k', alpha=0.7)
    axs[1, 0].bar(indices + width/2, p_h_sorted, width, label='Predicted', color='b', alpha=0.7)
    axs[1, 0].set_title("Asperity Height Distribution", fontsize=14)
    axs[1, 0].set_xlabel("Asperity Index (Sorted)", fontsize=12)
    axs[1, 0].set_ylabel("Height Offset h [m]", fontsize=12)
    axs[1, 0].legend(fontsize=11)

    # --- 4. Asperity Shape Exponents ---
    axs[1, 1].bar(indices - width/2, t_n_sorted, width, label='Target', color='k', alpha=0.7)
    axs[1, 1].bar(indices + width/2, p_n_sorted, width, label='Predicted', color='b', alpha=0.7)
    
    axs[1, 1].axhline(y=1.0, color='r', linestyle=':', lw=2, label='Flat Punch (n=1.0)')
    axs[1, 1].axhline(y=2.0, color='grey', linestyle=':', lw=2, label='Hertzian (n=2.0)')
    
    axs[1, 1].set_title("Shape Exponent Distribution", fontsize=14)
    axs[1, 1].set_xlabel("Asperity Index (Sorted)", fontsize=12)
    axs[1, 1].set_ylabel("Exponent n [-]", fontsize=12)
    axs[1, 1].set_ylim(0.8, 3.2)
    axs[1, 1].legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig