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
        
        # Unpack INTENSIVE properties
        pressure_gt = sample_data["pressure_gt"]
        alpha_gt = sample_data["alpha_gt"]
        alpha_nn_analytical = sample_data["alpha_nn_analytical"]
        pressure_bem = sample_data["pressure_bem"]
        alpha_bem = sample_data["alpha_bem"]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Ground Truth (Target)
        ax.plot(pressure_gt, alpha_gt, 'k-', lw=3, label="Target (GT)")
        
        # NN Analytical Reconstruction
        ax.plot(pressure_gt, alpha_nn_analytical, 'b--', lw=2, label="NN Analytical")
        
        # Tamaas BEM Verification
        if pressure_bem is not None and alpha_bem is not None:
            # Drop NaN values where BEM failed to converge
            valid = ~np.isnan(alpha_bem)
            ax.plot(pressure_bem[valid], alpha_bem[valid], 'ro', markersize=6, 
                    label="Tamaas BEM (Validation)", markeredgecolor='k')

        ax.set_title(f"Intensive Contact Mechanics: {key}")
        ax.set_xlabel("Nominal Pressure, $P$ [Pa]")
        ax.set_ylabel("Contact Fraction, $\\alpha$ [-]")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = f"plots/tamaas_val_{key}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
        plt.close()

if __name__ == "__main__":
    plot_validation_results()