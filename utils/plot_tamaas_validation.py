import numpy as np
import matplotlib.pyplot as plt
import os

def plot_validation_results(npz_path="data/paper_validation_data.npz"):
    if not os.path.exists(npz_path):
        print(f"Error: Could not find {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    os.makedirs("plots", exist_ok=True)

    for key in data.keys():
        sample_data = data[key].item()
        
        pressure_gt = sample_data["pressure_gt"]
        alpha_gt = sample_data["alpha_gt"]
        alpha_nn_analytical = sample_data["alpha_nn_opt_analytical"] 
        pressure_bem = sample_data["pressure_bem"]
        alpha_bem = sample_data["alpha_bem"]
        
        # Extract the saved surface data
        surface_bem = sample_data.get("surface_bem", None)
        L_bem = sample_data.get("L_bem", 1.0)

        alpha_gt_clean = np.where(alpha_gt == -1.0, np.nan, alpha_gt)
        alpha_nn_clean = np.where(alpha_nn_analytical == -1.0, np.nan, alpha_nn_analytical)

        # Create a 1x2 layout: 2D plot on the left, 3D on the right
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)

        # --- LEFT: Mechanics Plot ---
        ax1.plot(pressure_gt, alpha_gt_clean, 'k-', lw=4, label="Target (GT)", alpha=0.7)
        ax1.plot(pressure_gt, alpha_nn_clean, 'b--', lw=2.5, label="NN Optimizer (Analytical)", alpha=0.9)
        
        if pressure_bem is not None and alpha_bem is not None:
            valid = ~np.isnan(alpha_bem)
            ax1.plot(pressure_bem[valid], alpha_bem[valid], 'ro', markersize=6, 
                    label="Tamaas BEM (Validation)", markeredgecolor='k', zorder=5)

        ax1.set_xlabel(r"$P^*$", fontsize=12)
        ax1.set_ylabel(r"$\alpha = A/A_{tot}$", fontsize=12)
        
        max_p_val = np.nanmax(np.where(~np.isnan(alpha_gt_clean), pressure_gt, np.nan))
        if not np.isnan(max_p_val):
            ax1.set_xlim(0, max_p_val * 1.05)

        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # --- RIGHT: 3D Topography Plot ---
        if surface_bem is not None:
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            
            # Generate the X, Y coordinates for the grid
            N_pixels = surface_bem.shape[0]
            x = np.linspace(0, L_bem, N_pixels)
            y = np.linspace(0, L_bem, N_pixels)
            X, Y = np.meshgrid(x, y, indexing='ij')

            # Plot the surface
            surf = ax2.plot_surface(X, Y, surface_bem, cmap='coolwarm', 
                                    linewidth=0, antialiased=True, alpha=0.9)
            
            ax2.set_title(f"Predicted Surface", fontsize=14)
            ax2.set_zlabel("Height", fontsize=10)
            
            # Add a color bar
            fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
            
            # Optional: Adjust the viewing angle for better presentation
            ax2.view_init(elev=30, azim=45)

        save_path = f"plots/tamaas_val_{key}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    plot_validation_results()