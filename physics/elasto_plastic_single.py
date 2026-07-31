import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
import os
import time

try:
    import tamaas as tm
    from tamaas.nonlinear_solvers import DFSANECXXSolver
    from tamaas.dumpers import H5Dumper
except ImportError:
    tm = None


def kappa(gamma):
    return np.sqrt(np.pi) * sc.gamma(gamma / 2 + 1) / sc.gamma((gamma + 1) / 2)


def get_surface_topo(L, N_ep, c, gamma):
    x = np.linspace(0, L, N_ep, endpoint=False, dtype=np.float64)
    y = np.linspace(0, L, N_ep, endpoint=False, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    cx, cy = L / 2.0, L / 2.0
    r_sq = (xx - cx)**2 + (yy - cy)**2
    return - c * np.power(r_sq, gamma / 2.0)


def plot_surface_topography(topography, xx, yy, L):
    """
    Plot the surface topography using a 3D surface plot.
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        xx, yy, topography, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Topography', fontweight='bold')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Height')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def run_elastic_sweep(pressures_true, L_true, N_ep, c, gamma, E_star):
    print("\n--- Running Purely Elastic Sweep (Narrow Domain) ---")
    model = tm.Model(tm.model_type.basic_2d, [L_true, L_true], [N_ep, N_ep])
    model.nu = 0.3
    model.E = E_star * (1 - model.nu**2)

    surface_topo = get_surface_topo(L_true, N_ep, c, gamma)
    solver = tm.PolonskyKeerRey(model, surface_topo, 1e-9)

    alphas_true = []
    for p_mean in pressures_true:
        solver.solve(p_mean * E_star)
        mask = model.traction > 0
        a_frac = np.sum(mask) / (N_ep * N_ep)
        alphas_true.append(a_frac)

        if p_mean == pressures_true[-1]:
            # debug plot displacement and topography on a single figure, use heatmap for displacement
            uz_far_feild = model.displacement[0, 0]
            uz_plot = -(model.displacement[:, :] - uz_far_feild)

            pressure = model.traction[:, :]

        print(
            f"  > Solved P_mean = {p_mean:.4e} | Alpha = {a_frac:.4f}")

    return alphas_true, uz_plot, pressure


def run_plastic_sweep(pressures_true, L_true, L_z, N_ep, N_z, c, gamma, E_star, sigma_y_ratio):
    print(
        f"\n--- Running Elasto-Plastic Sweep (Narrow Domain | sigma_y/E*={sigma_y_ratio}) ---")
    model = tm.Model(tm.model_type.volume_2d, [
                     L_z, L_true, L_true], [N_z, N_ep, N_ep])
    model.nu = 0.3
    model.E = E_star * (1 - model.nu**2)

    surface_topo = get_surface_topo(L_true, N_ep, c, gamma)
    sigma_y = sigma_y_ratio * E_star

    material = tm.materials.IsotropicHardening(
        model, sigma_y=sigma_y, hardening=0.05 * model.E)
    residual = tm.Residual(model, material)

    epsolver = DFSANECXXSolver(residual)
    csolver = tm.PolonskyKeerRey(model, surface_topo, 1e-7)
    epic = tm.AndersonMixing(csolver, epsolver, 1e-5)

    dumper = H5Dumper(
        f"plastic_sim_gamma{gamma}", "displacement", "traction", "plastic_strain")
    model.addDumper(dumper)

    alphas_true = []
    p_fields_2d = []
    deltas_true = []

    t0 = time.time()
    for p_mean in pressures_true:
        epic.solve(p_mean * E_star)

        mask = model.traction[:, :, -1] > 0
        a_frac = np.sum(mask) / (N_ep * N_ep)
        alphas_true.append(a_frac)

        p_fields_2d.append(np.copy(model.traction))
        model.dump()

        if p_mean == pressures_true[-1]:
            # 1. Extract 2D Normal fields from the 3D/4D Volume Arrays
            p_normal = model.traction[:, :, -1]
            u_z_surface = model.displacement[0, :, :, -1]

            # 2. Shift displacement so the far-field (corner) is 0.0
            # We flip the sign so the compression into the material is negative
            u_far_field = u_z_surface[0, 0]
            u_plot = -(u_z_surface - u_far_field)

        print(
            f"  > Solved P_mean = {p_mean:.4e} | Alpha = {a_frac:.4f}")

    print(f"Plastic sweep finished in {time.time()-t0:.2f} seconds.")
    return alphas_true, u_plot, p_normal


def validate_yielding(gamma=1.8, R=1.0, delta_max_ratio=0.02, sigma_y_ratio=0.015):
    if tm is None:
        raise ImportError("Tamaas library not found.")

    tm.set_log_level(tm.LogLevel.error)
    tm.initialize(4)

    E_star = 1.0
    delta_max = R * delta_max_ratio
    c = 1.0 / (2.0 * R**(gamma - 1))

    print(
        f"Validating delta_max = {delta_max:.4e} | sigma_y/E* = {sigma_y_ratio:.4f} | gamma = {gamma:.2f}")
    # 1. Size domains
    gamma_max = 4.0
    k_val_max = kappa(gamma_max)
    a_for_L_max = (delta_max / (k_val_max * c))**(1.0 / gamma_max)
    L_max = 4.0 * a_for_L_max

    k_val = kappa(gamma)
    a_true_max = (delta_max / (k_val * c))**(1.0 / gamma)
    L_true = 4.0 * a_true_max

    print(
        f"ratio L_true/L_max = {L_true/L_max:.4f} | a_true_max = {a_true_max:.4e}")

    N_ep, N_z = 64, 32
    L_z = L_true / 2

    # 2. Calculate true local pressures
    F_max = E_star * (2 * gamma / (gamma + 1)) * k_val * \
        c * (a_true_max**(gamma + 1))
    P_max_true = F_max / (L_true**2 * E_star)

    n_steps = 10
    pressures_true = np.linspace(0, P_max_true, n_steps + 1)[1:]

    # 3. Analytical targets
    deltas_anl = np.linspace(0, delta_max, 100)
    a_anl = (deltas_anl / (k_val * c))**(1.0 / gamma)
    F_anl = E_star * (2 * gamma / (gamma + 1)) * \
        k_val * c * (a_anl**(gamma + 1))
    p_star_anl = F_anl / (L_max**2 * E_star)
    alpha_anl = (np.pi * a_anl**2) / (L_max**2)

    area_ratio = (L_true / L_max)**2
    print(f"  > Area ratio (L_true/L_max)^2 = {area_ratio:.4f}")
    data_path = "data/ep_validation_data.npz"

    # # degub plot surface topography
    # surface_topo = get_surface_topo(L_true, N_ep, c, gamma)
    # x = np.linspace(0, L_true, N_ep, endpoint=False, dtype=np.float64)
    # y = np.linspace(0, L_true, N_ep, endpoint=False, dtype=np.float64)
    # xx, yy = np.meshgrid(x, y, indexing='ij')
    # plot_surface_topography(surface_topo, xx, yy, L_true)

    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        alphas_el = data['alphas_el']
        alphas_ep = data['alphas_ep']
        uz_el = data['uz_el']
        uz_ep = data['uz_ep']
        pressure_el = data['pressure_el']
        pressure_ep = data['pressure_ep']
        pressures_star = data['pressures_star']

    else:
        alphas_true_el, uz_el, pressure_el = run_elastic_sweep(
            pressures_true, L_true, N_ep, c, gamma, E_star)

        alphas_true_ep, uz_ep, pressure_ep = run_plastic_sweep(
            pressures_true, L_true, L_z, N_ep, N_z, c, gamma, E_star, sigma_y_ratio)

        # # Placeholder for plastic sweep results
        # alphas_true_ep = np.zeros_like(pressures_true)
        # # Placeholder for pressure fields
        # p_fields_2d = np.zeros((len(pressures_true), N_ep, N_ep))

        # Scale parameters
        alphas_el = np.array(alphas_true_el) * area_ratio
        alphas_ep = np.array(alphas_true_ep) * area_ratio
        pressures_star = pressures_true * area_ratio

        # Insert origins
        alphas_el = np.insert(alphas_el, 0, 0.0)
        alphas_ep = np.insert(alphas_ep, 0, 0.0)
        pressures_star = np.insert(pressures_star, 0, 0.0)

        os.makedirs("data", exist_ok=True)
        np.savez(data_path, pressures_star=pressures_star, alphas_el=alphas_el,
                 alphas_ep=alphas_ep, uz_el=uz_el, uz_ep=uz_ep, pressure_el=pressure_el, pressure_ep=pressure_ep)

    # 5. Plotting
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(6, 4))

    plt.plot(p_star_anl, alpha_anl, color="#ADD8E6",
             lw=5, label='Analytical')
    plt.plot(pressures_star, alphas_el, ls=':', color='#D55E00',
             lw=2, marker='s', label='BEM Elastic')
    plt.plot(pressures_star, alphas_ep, ls='--', color="#940404", lw=2,
             marker='o', label=f'BEM Elasto-Plastic')

    plt.title(
        f'Yielding Deviation for Sharp Asperity ($\\gamma={gamma}$)', fontweight='bold')
    plt.xlabel(r'Nominal Pressure $P^*$')
    plt.ylabel(r'Contact Area Fraction $\alpha$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 1. Compute shared limits for accurate color mapping
    vmin_uz = min(uz_el.min(), uz_ep.min())
    vmax_uz = max(uz_el.max(), uz_ep.max())

    vmin_p = min(pressure_el.min(), pressure_ep.min())
    vmax_p = max(pressure_el.max(), pressure_ep.max())

    # 2. Use layout='constrained' to prevent colorbars from distorting plot sizes!
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True,
                            sharey=True, layout='constrained')

    # --- TOP ROW: DISPLACEMENT ---
    im_uz1 = axs[0, 0].imshow(uz_el, cmap='viridis',
                              origin='lower', vmin=vmin_uz, vmax=vmax_uz)
    im_uz2 = axs[0, 1].imshow(uz_ep, cmap='viridis',
                              origin='lower', vmin=vmin_uz, vmax=vmax_uz)

    axs[0, 0].set_title('Elastic', fontweight='bold', fontsize=12)
    axs[0, 1].set_title('Elasto-Plastic', fontweight='bold', fontsize=12)

    # Pass the entire row to the 'ax' parameter so the colorbar is centered and shares space evenly
    fig.colorbar(im_uz1, ax=axs[0, :], location='right', shrink=0.9, aspect=20,
                 label='$(u_z - u_{z, \mathrm{corner}})/R$')

    # --- BOTTOM ROW: TRACTION ---
    im_p1 = axs[1, 0].imshow(pressure_el, cmap='inferno',
                             origin='lower', vmin=vmin_p, vmax=vmax_p)
    im_p2 = axs[1, 1].imshow(pressure_ep, cmap='inferno',
                             origin='lower', vmin=vmin_p, vmax=vmax_p)

    # Shared Colorbar for Bottom Row
    fig.colorbar(im_p1, ax=axs[1, :], location='right', shrink=0.9, aspect=20,
                 label='$P/E^*$')

    # Clean up redundant axis labels
    for ax in axs[1, :]:
        ax.set_xlabel('X-axis')
    for ax in axs[:, 0]:
        ax.set_ylabel('Y-axis')

    # IMPORTANT: Do not call plt.tight_layout() when using layout='constrained'
    plt.savefig(f"plots/plastic_sweep_gamma_{gamma}_final.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    validate_yielding()
