import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# 1. Load Data
data = np.load("./data/paper_validation_data.npz", allow_pickle=True)

# 2. Setup Plot
fig, ax = plt.subplots(figsize=(7, 6))
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'}) # Paper-quality fonts

# Using a specific qualitative palette
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e'] 
# Blue, Red, Green, Purple, Orange

# 4. Loop through samples
for i, key in enumerate(data.files):
    sample = data[key].item()
    
    # Pick color for this sample (cycle if more than 5 samples)
    c = colors[i % len(colors)]
    
    # Clean up the label name (e.g., "sample_0" -> "Sample A")
    label_name = f"Sample {i+1}"
    
    # A. Plot Target (Ground Truth) -> Solid Line
    # We only label the Target to keep the legend clean(er)
    ax.plot(sample['load_gt'], sample['area_gt'], 
            color=c, linestyle='-', linewidth=2.5, alpha=0.9, 
            label=label_name)

    # B. Plot NN Prediction -> Dashed Line
    ax.plot(sample['load_gt'], sample['area_nn_analytical'], 
            color=c, linestyle='--', linewidth=2.0, alpha=0.9)

    # C. Plot Tamaas (BEM) -> Points with white outline (pops nicely)
    if sample['area_bem'] is not None:
        ax.plot(sample['load_bem'], sample['area_bem'], 
                marker='o', linestyle='None', markersize=7, 
                color=c, markeredgecolor='white', markeredgewidth=1.0)

# 5. Styling
ax.set_xlabel(r"Normal Load $F$ [N]")
ax.set_ylabel(r"Real Contact Area $A$ [mm$^2$]")
# ax.set_title("Validation: Neural Network vs. BEM vs. Ground Truth")
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
j
# Legend 1: Samples (Colors)
# We already added labels to the Target plots, so standard legend handles colors
legend1 = ax.legend(loc='upper left', title="Samples")
ax.add_artist(legend1)

# Legend 2: Methods (Styles)
line_target = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2.5, label='Target (GT)')
line_nn     = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2.0, label='NN Prediction')
marker_bem  = mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                          markersize=7, markeredgecolor='white', label='Tamaas (BEM)')

ax.legend(handles=[line_target, line_nn, marker_bem], loc='lower right', title="Method")
plt.savefig("tamaas_validation_plot.png", dpi=300)
plt.tight_layout()
plt.show()