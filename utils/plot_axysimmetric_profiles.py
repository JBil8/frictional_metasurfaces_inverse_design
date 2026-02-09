import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_asperity_profiles():
    # 1. Setup Data
    r = np.linspace(-1, 1, 1000)
    
    # Exponents
    gammas = [1, 2, 4, 8, 100]  
    
    # Labels with fancy math text
    labels = {
        1:   r'$\gamma=1$ (Cone)', 
        2:   r'$\gamma=2$ (Hertz)', 
        4:   r'$\gamma=4$', 
        8:   r'$\gamma=8$', 
        100: r'$\gamma \to \infty$ (Flat)'
    }
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 1.0
    })

    # 3. Colors: Dark Gray (Cone) -> Light Gray (Flat)
    # Start at 0.3 (Dark) and go to 0.85 (Light)
    gray_values = np.linspace(0.3, 0.85, len(gammas))
    colors = [cm.gray(g) for g in gray_values]
    
    # We define the order of plotting: Biggest Volume (Flat) -> Smallest Volume (Cone)
    # This ensures the smaller shapes are drawn ON TOP of the larger ones.
    # Flat punch covers the whole box. Cone is just the V in the middle.
    plot_order = gammas[::-1] # [100, 8, 4, 2, 1]
    plot_colors = colors[::-1] # Light -> Dark

    z_top = 1.0

    # 4. Plot Loop
    for i, gamma in enumerate(plot_order):
        # Calculate Profile z = |r|^gamma
        z = np.abs(r)**gamma
        
        # Color & Text Color
        fill_color = plot_colors[i]
        
        # Determine text color based on background darkness
        # If gray value < 0.5 (Dark), use White text. Else Black.
        # We need to map back to the gray value. 
        current_gray = gray_values[-(i+1)]
        text_color = 'white' if current_gray < 0.55 else 'black'

        # Plot Outline
        ax.plot(r, z, color='black', linewidth=1.2, zorder=i+10)
        
        # Fill Area (Material is above the curve, up to z=1)
        ax.fill_between(r, z, z_top, color=fill_color, alpha=1.0, zorder=i)

        # --- Label Placement Logic ---
        # We place labels on the right side (r > 0)
        # if gamma == 1: 
        #     # Cone: Aligned with the slope
        #     ax.text(0.15, 0.8, labels[gamma], color=text_color, 
        #             ha='center', va='center', rotation=0, fontweight='bold')
        
        # elif gamma == 2: 
        #     # Hertz: Slightly lower
        #     ax.text(0.72, 0.65, labels[gamma], color=text_color, 
        #             ha='center', va='center', rotation=38, fontweight='bold')
            
        # elif gamma == 4:
        #     ax.text(0.83, 0.53, labels[gamma], color=text_color, 
        #             ha='center', va='center', rotation=55, fontsize=10)
            
        # elif gamma == 8:
        #     ax.text(0.91, 0.35, labels[gamma], color=text_color, 
        #             ha='center', va='center', rotation=75, fontsize=10)
            
        # elif gamma == 100: 
        #     # Flat: Bottom Center
        #     ax.text(0.0, 0.08, labels[gamma], color=text_color, 
        #             ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 5. Styling
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 1.0)

    ax.set_aspect('equal', adjustable='box')

    # Clean Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Labels
    ax.set_xlabel(r'$r$', fontsize=12)
    ax.set_ylabel(r'$h$', fontsize=12)
    
    # Ticks
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['1', '0', '1'])
    ax.set_yticks([]) # Hide Y ticks

    plt.tight_layout()
    plt.savefig('asperity_shapes_final.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_asperity_profiles()