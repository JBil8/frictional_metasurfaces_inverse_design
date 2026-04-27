import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction(t_p, t_a, t_s, p_p, p_a, p_s, t_params, p_params, epoch,
                        t_p_max, t_a_max, p_p_max, p_a_max, gamma_min, gamma_max):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Title displaying the absolute magnitude validation
    title = f"Epoch {epoch} | Target P*_max: {t_p_max:.2e} vs Pred P*_max: {p_p_max:.2e}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # 2. Dynamic X-axis limit to accommodate the longest curve
    max_x = max(t_p[-1], p_p[-1]) * 1.05

    # --- 1. Contact Fraction vs Pressure ---
    axs[0, 0].plot(t_p, t_a, 'k-', lw=3, label='Target')
    axs[0, 0].plot(p_p, p_a, 'b--', lw=2.5, label='Predicted')
    axs[0, 0].set_xlim(0, max_x)
    axs[0, 0].set_title("Contact Area vs Load (α vs P*)", fontsize=14)
    axs[0, 0].set_ylabel("Contact Fraction α", fontsize=12)
    axs[0, 0].legend()

    # --- 2. Intensive Stiffness vs Pressure ---
    axs[0, 1].plot(t_p, t_s, 'k-', lw=3, label='Target')
    axs[0, 1].plot(p_p, p_s, 'b--', lw=2.5, label='Predicted')
    axs[0, 1].set_xlim(0, max_x)
    axs[0, 1].set_title("Topological Stiffness (S*)", fontsize=14)
    axs[0, 1].set_ylabel("Stiffness S*", fontsize=12)
    axs[0, 1].legend()

    # Use scientific notation for x-axis due to small P* values
    for ax in [axs[0, 0], axs[0, 1]]:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.grid(True, linestyle='--', alpha=0.4)

    # --- Parameter Extraction & Sorting (Unchanged) ---
    n_asp = len(t_params) // 2
    t_n, t_h = t_params[:n_asp], t_params[n_asp:]
    p_n, p_h = p_params[:n_asp], p_params[n_asp:]

    t_sort_idx = np.argsort(t_h)
    p_sort_idx = np.argsort(p_h)

    indices = np.arange(n_asp)
    width = 0.35
    max_delta = 0.02

    # --- 3. Asperity Height Distribution ---
    axs[1, 0].bar(indices - width/2, t_h[t_sort_idx]/max_delta, width,
                  label='Target', color='k', alpha=0.7)
    axs[1, 0].bar(indices + width/2, p_h[p_sort_idx]/max_delta, width,
                  label='Predicted', color='b', alpha=0.7)
    axs[1, 0].set_title("Asperity Height Distribution", fontsize=14)
    axs[1, 0].set_ylabel(r"$h/\Delta_{max}$", fontsize=12)
    axs[1, 0].legend(fontsize=11)
    axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.4)

    # --- 4. Shape Exponent Distribution ---
    axs[1, 1].bar(indices - width/2, t_n[t_sort_idx], width,
                  label='Target', color='k', alpha=0.7)
    axs[1, 1].bar(indices + width/2, p_n[p_sort_idx], width,
                  label='Predicted', color='b', alpha=0.7)

    # Dynamically draw the boundary lines based on your config
    axs[1, 1].axhline(y=gamma_min, color='r', linestyle=':',
                      lw=2, label=f'Min ({gamma_min})')
    axs[1, 1].axhline(y=gamma_max, color='g', linestyle=':',
                      lw=2, label=f'Max ({gamma_max})')

    axs[1, 1].set_title("Shape Exponent Distribution", fontsize=14)
    axs[1, 1].set_ylabel("Exponent γ [-]", fontsize=12)

    # Dynamically scale the Y-axis with a 10% visual buffer
    buffer = (gamma_max - gamma_min) * 0.1
    axs[1, 1].set_ylim(gamma_min - buffer, gamma_max + buffer)

    axs[1, 1].legend(fontsize=11)
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
