import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_surface(grid_res=1500):
    R = 1.0
    spacing = 2.2 * R
    L = 3 * spacing

    x = np.linspace(-L/2, L/2, grid_res)
    y = np.linspace(-L/2, L/2, grid_res)
    X, Y = np.meshgrid(x, y)

    # Initialize the "floor"
    Z = np.full_like(X, -1.2)

    ns = np.linspace(1.0, 3.0, 9)

    idx = 0
    for i in range(3):
        for j in range(3):
            cx = (j - 1) * spacing
            cy = (1 - i) * spacing
            n = ns[idx]

            r_sq = (X - cx)**2 + (Y - cy)**2

            max_r_sq = (R**2) * (1.2 ** (2 / n))

            # Create a mask of only the pixels inside this specific asperity's radius
            mask = r_sq < max_r_sq

            local_z = -((r_sq[mask] / (R**2)) ** (n / 2))

            Z[mask] = np.maximum(Z[mask], local_z)
            idx += 1

    np.savez_compressed("data/surface_grid.npz", Z=Z)

    return X, Y, Z


def plot_surface():
    print("Generating Optimized Surface...")
    X, Y, Z = generate_surface(grid_res=1000)

    plt.rcParams.update({'font.family': 'serif', 'figure.dpi': 300})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('Grays', 256)  # Use a high-res colormap
    # shift the colormap so that there is no white at the bottom
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'shifted_grays',
        [(0.9, 0.9, 0.9), (0.0, 0., 0.)],
        N=256
    )
    ax.plot_surface(X, Y, Z,
                    rcount=150, ccount=150,
                    cmap=cmap, linewidth=0,
                    antialiased=True, alpha=1.0,
                    shade=True)

    ax.set_zlim(-1.2, 0.5)
    ax.set_axis_off()
    ax.view_init(elev=35, azim=-45)
    ax.set_box_aspect((1, 1, 0.4))

    plt.tight_layout()
    plt.savefig("plots/surface_plot.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    plot_surface()
