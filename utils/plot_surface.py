import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm # Not needed for solid colors

# --- Helper function from your code ---
def generate_surface_for_plot(sample_data, grid_res=2000):
    """
    Reconstructs the 3D surface from the saved parameters (n, h).
    """
    # Try to get ground truth parameters, fallback to defaults for visualization if missing
    ns = sample_data.get('n_gt', np.ones(16)*2.0)
    hs = sample_data.get('h_gt', np.linspace(0, 0.1, 16))
    
    # Grid setup
    L = 4.0 # Arbitrary visual scale
    x = np.linspace(-L/2, L/2, grid_res)
    y = np.linspace(-L/2, L/2, grid_res)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X) - 1.0 # Initialize deep
    
    # Place asperities in a 4x4 grid
    grid_dim = int(np.sqrt(len(ns)))
    if grid_dim == 0: grid_dim = 1 # Safety check
    spacing = L / grid_dim
    
    for i in range(len(ns)):
        row = i // grid_dim
        col = i % grid_dim
        cx = -L/2 + (col + 0.5) * spacing
        cy = -L/2 + (row + 0.5) * spacing
        
        # Local shape: z = -h - |r|^n
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Width w ~ spacing/2 to avoid overlap
        w = spacing * 0.4 
        
        # Protect against negative base for fractional exponents
        safe_r = np.maximum(r, 1e-9)
        local_z = -hs[i] - (safe_r/w)**ns[i]
        
        # Union (Max height)
        Z = np.maximum(Z, local_z)
        
    return X, Y, Z

def plot_3d_surfaces_colored():
    # 1. Load Data
    try:
        data = np.load("./data/paper_validation_data.npz", allow_pickle=True)
        keys = list(data.files)[:3] # Grab first 3 samples
    except FileNotFoundError:
        print("Error: './data/paper_validation_data.npz' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    fig = plt.figure(figsize=(15, 5))
    
    # Paper quality settings
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

    # Define specific colors to match the validation plot
    # Sample 1: Blue, Sample 2: Red, Sample 3: Green
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    for i, key in enumerate(keys):
        sample = data[key].item()
        
        # Generate 3D Mesh
        X, Y, Z = generate_surface_for_plot(sample)

        # Plot
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Surface Plot
        # We clip Z for better visualization (don't show infinite depth)
        zlim_inf = -0.4
        Z_plot = np.maximum(Z, zlim_inf)
        
        # Get the color for this sample
        sample_color = colors[i % len(colors)]
        
        # Use 'color' instead of 'cmap'. 'shade=True' helps with 3D perception.
        surf = ax.plot_surface(X, Y, Z_plot, color=sample_color, 
                               linewidth=0, antialiased=True, alpha=0.9, shade=True)
        
        # Styling
        # Color the title to match the surface for clarity
        # ax.set_title(f"Design {i+1}", fontsize=12, fontweight='bold', color=sample_color)
        ax.set_zlim(zlim_inf, 0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=45, azim=45) # Nice angle
        
        # Remove panes for clean look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_aspect('equal')
        ax.grid(False)

    plt.tight_layout()
    save_path = 'surface_topographies_3d_colored.png'
    plt.savefig(save_path, dpi=300)
    print(f"3D Surface plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_3d_surfaces_colored()