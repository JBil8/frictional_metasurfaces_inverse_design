import torch
from physics.differentiable import AxisymmetricContactLayer

def get_theoretical_limits(cfg, device):
    """
    Calculates the ABSOLUTE maximum Load, Area, and Stiffness (dF/dA)
    possible for the given physics configuration (N pseudo-flat punches).
    """
    phys = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)

    n_asp = cfg['physics']['n_asperities']
    R = cfg['physics']['radius']
    max_d = cfg['physics']['max_delta_ratio'] * R

    # 1. Create the "Ultimate Wall" (All Pseudo-Flat Punches touching at h=0)
    h_wall = torch.zeros(1, n_asp).to(device)
    n_wall = torch.ones(1, n_asp).to(device) * 3.0  # Max Exponent
    w_wall = torch.ones(1, n_asp).to(device) * 2.0 * R

    # 2. Indent fully
    # We use the actual step count from config to ensure perfectly matching arrays
    steps = cfg['data']['n_steps']
    ind = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)

    with torch.no_grad():
        # CHANGED: Unpack all 3 outputs from the updated physics layer
        l_max, a_max, df_da_max = phys(h_wall, n_wall, w_wall, ind)

    return {
        "max_load": l_max.max().item(),
        "max_area": a_max.max().item(),
        # CHANGED: Directly take the max of the analytical derivative
        "max_stiff": df_da_max.max().item() 
    }