import os
import matplotlib.pyplot as plt

# use latex for math rendering


def read_metric(filepath):
    """Helper to parse the space-separated MLflow metric files."""
    epochs, values = [], []
    if not os.path.exists(filepath):
        print(f"Warning: File not found -> {filepath}")
        return epochs, values
        
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                values.append(float(parts[1]))
                epochs.append(int(parts[2]))
    return epochs, values

def plot_training_dynamics(metrics_dir="."):
    print("Generating 2x2 Training Dynamics Grid...")
    
    # 1. Read all the data
    ep_train, val_train = read_metric(os.path.join(metrics_dir, "train_loss"))
    ep_val, val_val = read_metric(os.path.join(metrics_dir, "val_loss"))
    ep_k, val_k = read_metric(os.path.join(metrics_dir, "k_steepness"))
    ep_lam, val_lam = read_metric(os.path.join(metrics_dir, "lambda_param"))
    ep_lr, val_lr = read_metric(os.path.join(metrics_dir, "learning_rate"))
    
    # 2. Setup the 2x2 Grid with a shared X-axis
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})
    fig, axs = plt.subplots(2, 2, figsize=(6, 5.5), sharex=True)
    
    # --- Top Left: The Losses ---
    axs[0, 0].plot(ep_train, val_train, color='black', lw=2.5, label='Train Loss', linestyle='-')
    axs[0, 0].plot(ep_val, val_val, color='black', lw=2.5, label='Val Loss', linestyle=':')
    # axs[0, 0].set_title("Reconstruction Loss", fontweight='bold')
    axs[0, 0].set_ylabel(r"$\mathcal{L}$")
    axs[0, 0].legend(loc='upper right', framealpha=0.9)
    # Optional: If your initial loss is massive, uncomment the next line
    # axs[0, 0].set_yscale('log') 

    # --- Top Right: Kappa (Homotopy Steepness) ---
    axs[0, 1].plot(ep_k, val_k, color='black', lw=2.5)
    # axs[0, 1].set_title("Physics Resolution (Kappa)", fontweight='bold')
    axs[0, 1].set_ylabel(r"$\kappa$")
    axs[0, 1].set_yscale('log') # Log scale because it goes from 10^3 to 10^5

    # --- Bottom Left: Lambda (Parameter Regularization) ---
    axs[1, 0].plot(ep_lam, val_lam, color='black', lw=2.5)
    # axs[1, 0].set_title("Curriculum Weight (Lambda)", fontweight='bold')
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel(r"$\lambda$")
    axs[1, 0].set_ylim(-0.005, max(val_lam)*1.1 if val_lam else 0.1) # Keep zero anchored

    # --- Bottom Right: Learning Rate ---
    axs[1, 1].plot(ep_lr, val_lr, color='black', lw=2.5)
    # axs[1, 1].set_title("Learning Rate Annealing", fontweight='bold')
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Learning Rate")
    axs[1, 1].set_yscale('log') # Standard for LR plots
    
    # 3. Clean Formatting for all subplots
    for ax in axs.flatten():
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i, ax in enumerate(axs.flat):
            ax.text(-0.3, 1.0, f"({chr(97+i)})", transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    
    # Save the figure
    save_path = 'plots/training_dynamics_grid.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved successfully to {save_path}")
    plt.show()


if __name__ == "__main__":
   
    DIRECTORY_PATH = "./data/metrics" 
    
    plot_training_dynamics(DIRECTORY_PATH)