import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(p_grid, t_alpha, t_s, p_alpha, p_s, t_params, p_params, epoch=None):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    title = f"Epoch {epoch} Reconstruction" if epoch is not None else "Model Prediction"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    # Helper to clean data: Replace padding (-1.0) with NaN for clean plotting
    def clean_for_plot(data):
        # We assume any value < -0.5 is padding based on your current plot appearance
        # Using -0.5 is safer if your normalization pushes -1.0 to different values
        data_cp = data.copy()
        data_cp[data <= -0.1] = np.nan 
        return data_cp

    # Clean the curves
    t_a_plot = clean_for_plot(t_alpha)
    p_a_plot = clean_for_plot(p_alpha)
    t_s_plot = clean_for_plot(t_s)
    p_s_plot = clean_for_plot(p_s)

    # Determine the x-axis limit: Find the last index where target is NOT nan
    # This zooms the plot into "where the physics happens"
    valid_indices = np.where(~np.isnan(t_a_plot))[0]
    if len(valid_indices) > 0:
        # Give 10% breathing room beyond the target end
        max_idx = min(len(p_grid) - 1, int(valid_indices[-1] * 1.1))
        x_limit = p_grid[max_idx]
    else:
        x_limit = p_grid[-1]

    # --- 1. Contact Fraction vs Pressure ---
    axs[0, 0].plot(p_grid, t_a_plot, 'k-', lw=3, label='Target')
    axs[0, 0].plot(p_grid, p_a_plot, 'b--', lw=2.5, label='Predicted')
    axs[0, 0].set_xlim(0, x_limit) # Zoom in
    axs[0, 0].set_title("Contact Area vs Load (α vs P*)", fontsize=14)
    axs[0, 0].set_ylabel("Contact Fraction α", fontsize=12)
    axs[0, 0].legend()

    # --- 2. Intensive Stiffness vs Pressure ---
    axs[0, 1].plot(p_grid, t_s_plot, 'k-', lw=3, label='Target')
    axs[0, 1].plot(p_grid, p_s_plot, 'b--', lw=2.5, label='Predicted')
    axs[0, 1].set_xlim(0, x_limit) # Zoom in
    axs[0, 1].set_title("Topological Stiffness (S*)", fontsize=14)
    axs[0, 1].set_ylabel("Stiffness S*", fontsize=12)
    axs[0, 1].legend()

    # --- Parameter Extraction & Sorting (Keep as is) ---
    n_asp = len(t_params) // 2
    t_n, t_h = t_params[:n_asp], t_params[n_asp:]
    p_n, p_h = p_params[:n_asp], p_params[n_asp:]
    
    t_sort_idx = np.argsort(t_h)
    p_sort_idx = np.argsort(p_h)
    
    indices = np.arange(n_asp)
    width = 0.35

    axs[1, 0].bar(indices - width/2, t_h[t_sort_idx], width, label='Target', color='k', alpha=0.7)
    axs[1, 0].bar(indices + width/2, p_h[p_sort_idx], width, label='Predicted', color='b', alpha=0.7)
    axs[1, 0].set_title("Asperity Height Distribution", fontsize=14)
    axs[1, 0].set_ylabel("Height Offset h [m]", fontsize=12)
    axs[1, 0].legend(fontsize=11)

    axs[1, 1].bar(indices - width/2, t_n[t_sort_idx], width, label='Target', color='k', alpha=0.7)
    axs[1, 1].bar(indices + width/2, p_n[p_sort_idx], width, label='Predicted', color='b', alpha=0.7)
    axs[1, 1].axhline(y=1.0, color='r', linestyle=':', lw=2, label='n=1.0')
    axs[1, 1].axhline(y=3.0, color='g', linestyle=':', lw=2, label='n=3.0')
    axs[1, 1].set_title("Shape Exponent Distribution", fontsize=14)
    axs[1, 1].set_ylabel("Exponent n [-]", fontsize=12)
    axs[1, 1].set_ylim(0.8, 3.2)
    axs[1, 1].legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig