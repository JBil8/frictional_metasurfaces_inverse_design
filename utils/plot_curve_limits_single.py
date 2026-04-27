import numpy as np
import matplotlib.pyplot as plt

def plot_design_space_loglog_shaded():
   
    # Normalized Load P from small (1e-3) to 1
    P = np.logspace(0, 2, 200)
    
    # --- Bottom Limit: Cone (Gamma = 1) ---
    A_cone = P ** 1.0
    
    # --- Middle Boundary: Hertz (Gamma = 2) ---
    A_hertz = P ** (2/3)
    
    # Cubic
    A_cubic = P ** 0.5

    # --- Top Limit: Flat Punch (Gamma -> infinity) ---
    A_flat = np.ones_like(P)
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    
    # Publication styling
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in', 'ytick.direction': 'in'
    })

    # 4. Plot Lines (The Skeleton)
    # Cone (Bottom)
    ax.loglog(P, A_cone, 'k-', linewidth=1.5, label=r'$\gamma=1$ (Cone)')
    
    # Hertz (Middle)
    ax.loglog(P, A_hertz, 'k--', linewidth=2.0, label=r'$\gamma=2$ (Hertz)')
    
    # Flat (Top)
    ax.loglog(P, A_cubic, 'k-.', linewidth=1.5, label=r'$\gamma=3$ (Cubic)')

    # 5. Shading (The Story)
    
    
    ax.fill_between(P, A_cone, A_hertz, color='gray', alpha=0.2)
    
    
    ax.text(1e-2, 2e-3, "Standard Roughness\n(Linearization Regime)", 
            color='dimgray', fontsize=10, rotation=42, ha='center', va='center')

    
    ax.fill_between(P, A_hertz, A_flat, color='firebrick', alpha=0.15)
    
    # Add Text for Red Zone
    ax.text(1e-2, 0.1, "Metainterfaces\n(Design Space)", 
            color='darkred', fontsize=10, fontweight='bold', rotation=10, ha='center', va='center')

    # 6. Formatting
    ax.set_xlabel(r'$P$', fontsize=12)
    ax.set_ylabel(r'$A$', fontsize=12)
    
    # Limits
    ax.set_xlim(1, 19)
    ax.set_ylim(1, 19) # slightly lower to show the cone tail
    
    # Clean Legend
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # plt.tight_layout()
    plt.savefig('design_space_loglog_final.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_design_space_loglog_shaded()