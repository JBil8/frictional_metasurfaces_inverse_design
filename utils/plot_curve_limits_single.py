import numpy as np
import matplotlib.pyplot as plt

def plot_design_space_loglog_shaded():
    # 1. Setup Data (Log Log Space)
    # Normalized Load P from small (1e-3) to 1
    P = np.logspace(0, 2, 200)
    
    # 2. Define Curves A ~ P^(2/(gamma+1))
    
    # --- Bottom Limit: Cone (Gamma = 1) ---
    # The limit of "infinite roughness" (Archard)
    # Slope = 1.0
    A_cone = P ** 1.0
    
    # --- Middle Boundary: Hertz (Gamma = 2) ---
    # The limit of "smooth sphere" (Convexity Limit)
    # Slope = 0.66...
    A_hertz = P ** (2/3)
    
    # Cubic
    A_cubic = P ** 2

    # --- Top Limit: Flat Punch (Gamma -> infinity) ---
    # The limit of "constant area"
    # Slope = 0
    A_flat = np.ones_like(P)
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    
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
    
    # GRAY REGION: Standard Roughness (Cone <-> Hertz)
    # This represents the "Archard" linearization effect.
    ax.fill_between(P, A_cone, A_hertz, color='gray', alpha=0.2)
    
    # Add Text for Gray Zone
    # Place it in the middle of the gray wedge
    ax.text(1e-2, 2e-3, "Standard Roughness\n(Linearization Regime)", 
            color='dimgray', fontsize=10, rotation=42, ha='center', va='center')

    # RED REGION: Metainterfaces (Hertz <-> Flat)
    # This represents your "Beyond Hertz" contribution.
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
    
    plt.tight_layout()
    plt.savefig('design_space_loglog_final.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_design_space_loglog_shaded()