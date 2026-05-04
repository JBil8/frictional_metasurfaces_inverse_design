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

    # Quartic
    A_quartic = P ** 0.25

    # --- Top Limit: Flat Punch (Gamma -> infinity) ---
    A_flat = np.ones_like(P)
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # Publication styling
    plt.rcParams.update({
        'font.size': 20, 'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in', 'ytick.direction': 'in'
    })

    # 4. Plot Lines (The Skeleton)
    # Cone (Bottom)
    ax.loglog(P, A_cone, 'k-', linewidth=2.5, label=r'$\gamma=1$ (Cone)')
    
    # Hertz (Middle)
    ax.loglog(P, A_hertz, 'k--', linewidth=3.0, label=r'$\gamma=2$ (Hertz)')
    
    # Flat (Top)
    # ax.loglog(P, A_cubic, 'k-.', linewidth=1.5, label=r'$\gamma=3$ (Cubic)')

    ax.loglog(P, A_quartic, 'k-.', linewidth=2.5, label=r'$\gamma=4$ (Quartic)')

    # 5. Shading and Text (The Narrative)
    
    # Standard Roughness Zone (Cone to Hertz)
    # ax.fill_between(P, A_cone, A_hertz, color='gray', alpha=0.2)
    # ax.text(3, 6, "Standard Roughness\n(Linearization Regime)", 
    #         color='dimgray', fontsize=16, rotation=38, ha='center', va='center')

    # Meta-Interface Zone (Hertz to Flat Punch)
    # A_flat is 1.0 everywhere, shading down to the absolute theoretical limit
    ax.fill_between(P, A_cone, A_flat, color='gray', alpha=0.15)
    # ax.text(8, 2.5, "Metainterfaces\n(New Design Space)", 
    #         color='darkred', fontsize=16, fontweight='bold', rotation=22, ha='center', va='center')

    # 6. Formatting
    ax.set_xlabel(r'Load $F$', fontsize=20)
    ax.set_ylabel(r'Area $A$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18, length=10, width=2)

    # Limits
    ax.set_xlim(1, 19)
    ax.set_ylim(1, 19) # slightly lower to show the cone tail
    
    # Clean Legend
    ax.legend(loc='upper left', frameon=False, fontsize=20)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/design_space_loglog_final.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_design_space_loglog_shaded()