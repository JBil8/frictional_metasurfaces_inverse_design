import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# --- Publication Formatting ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 17,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
})

def plot_category_3_panel(sample_data, category_name, save_dir="plots"):
    """Generates a 1x3 grid: P-alpha Curve | 3D Surface | 2D Pressure Heatmap."""
    print(f"Generating 3-panel figure for: {category_name}...")
    
    fig = plt.figure(figsize=(16, 5))
    
    paper_names = {
            "linear": "Linear Coulomb",
            "bilinear": "Bilinear Transition",
            "saturating": "Saturating",
            "bimodal": "Bimodal",
            "sparse": "Stratified",
            "lhs": "LHS",
            "random_sum": "Mixed",
            "wall": "Coplanar",
            "exiled": "Truncated"
    }

    # --- Data Extraction ---
    p_gt = sample_data["pressure_gt"].flatten()
    a_gt = sample_data["alpha_gt"].flatten()
    a_nn = sample_data["alpha_nn_opt_analytical"].flatten()
    p_bem = sample_data["pressure_bem"]
    a_bem = sample_data["alpha_bem"]
    surface_bem = sample_data.get("surface_bem", None)
    pressure_field = sample_data.get("pressure_field_bem", None)
    L_bem = sample_data.get("L_bem", 1.0)

    # save the surface topography as a npz file for later use
    # if surface_bem is not None:
    #     surface_save_path = os.path.join(save_dir, f"tamaas_surface_{category_name}.npz")
    #     np.savez(surface_save_path, surface=surface_bem, L=L_bem)
    #     print(f"  > Saved BEM surface topography to: {surface_save_path}")

    # Clean NaNs
    a_gt = np.where(a_gt == -1.0, np.nan, a_gt)
    a_nn = np.where(a_nn == -1.0, np.nan, a_nn)

    # --- PANEL 1: Macroscopic Curve ---
    ax1 = fig.add_subplot(1, 3, 1)
    C_GT, C_NN, C_BEM = '#333333', '#0072B2', '#D55E00'

    
    ax1.plot(p_gt, a_gt, color=C_GT, lw=3.5, label="Target")
    ax1.plot(p_gt, a_nn, color=C_NN, linestyle='--', lw=2.5, label="NN + Opt")
    
    if p_bem is not None and len(p_bem) > 0:
        valid = ~np.isnan(a_bem)
        ax1.plot(p_bem[valid], a_bem[valid], marker='o', linestyle='none', 
                 color=C_BEM, markersize=7, markeredgecolor='white', markeredgewidth=1.0,
                 label="Tamaas BEM", zorder=5)

    lookup_key = category_name.replace('sample_', '').lower()
    title_clean = paper_names.get(lookup_key, lookup_key.replace('_', ' ').title())
    ax1.set_title(f"Area-Load Evolution ({title_clean})", fontweight='bold', pad=15)
    ax1.set_xlabel('$P^*$')
    ax1.set_ylabel(r'$\alpha$')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax1.legend(loc='lower right', frameon=False)

    # --- PANEL 2: 3D Surface ---
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    if surface_bem is not None:
        N_pixels = surface_bem.shape[0]
        x = np.linspace(-L_bem/2, L_bem/2, N_pixels)
        y = np.linspace(-L_bem/2, L_bem/2, N_pixels)
        X, Y = np.meshgrid(x, y)

        surf = ax2.plot_surface(X, Y, surface_bem, cmap='Grays', 
                                linewidth=0, antialiased=True, alpha=1.0, shade=True,
                                rcount=100, ccount=100)
        ax2.set_title("Neural Surrogate Topography", fontweight='bold', pad=15)
        
        # Add matching spatial limits and labels
        ax2.set_xlim(-L_bem/2, L_bem/2)
        ax2.set_ylim(-L_bem/2, L_bem/2)
        ax2.set_xlabel("x/R", labelpad=10)
        ax2.set_ylabel("y/R", labelpad=10)
        # ax2.set_zlabel("z/R", labelpad=10)
        ax2.set_zticks([])  # Hide z-axis ticks for cleaner look
        
        # Clean up the 3D axis panes for publication (removes the gray background walls)
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('white')
        ax2.yaxis.pane.set_edgecolor('white')
        ax2.zaxis.pane.set_edgecolor('white')
        
        # Optional: reduce grid clutter
        ax2.grid(alpha=0.3)
        ax2.view_init(elev=35, azim=-75)
        ax2.set_box_aspect((1, 1, 0.4))
    else:
        ax2.text(0.5, 0.5, 0.5, "Surface Data Missing", ha='center', va='center')
        ax2.set_axis_off()

    # --- PANEL 3: 2D Pressure Heatmap ---
    ax3 = fig.add_subplot(1, 3, 3)
    if pressure_field is not None:
        im = ax3.imshow(pressure_field, cmap='Purples', extent=[-L_bem/2, L_bem/2, -L_bem/2, L_bem/2], origin='lower')
        ax3.set_title("Full-Field BEM Pressure ($P_{max}$)", fontweight='bold', pad=22)
        ax3.set_xlabel("x/R")
        ax3.set_ylabel("y/R")
        cbar = fig.colorbar(im, ax=ax3, shrink=0.75, pad=0.05)
        cbar.set_label('$P/E^*$', rotation=270, labelpad=15)
        
    else:
        ax3.text(0.5, 0.5, "Pressure Field Data Missing", ha='center', va='center')
        ax3.axis('off')

    # # Add subplot labels (a), (b), (c)
    # for i, ax in enumerate([ax1, ax2, ax3]):
    #         ax.text(-0.3, 1.0, f"({chr(97+i)})", transform=ax.transAxes, 
    #                 fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"tamaas_{category_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    save_directory = "plots"
    os.makedirs(save_directory, exist_ok=True)
    npz_path = "data/paper_validation_data.npz"
    
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        
        # Loop through every category saved in the dictionary
        for key in data.keys():
            sample_data = data[key].item()
            plot_category_3_panel(sample_data, category_name=key, save_dir=save_directory)
            
        print("\nAll 3-panel figures successfully generated.")
    else:
        print(f"Error: {npz_path} not found.")