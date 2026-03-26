import torch
from physics.differentiable import AxisymmetricContactLayer

def get_theoretical_limits(cfg, device):
    """
    Calculates the ABSOLUTE maximum Pressure, Contact Fraction (Alpha), 
    and Intensive Stiffness (dP/dAlpha) for the given physics configuration.
    """
    phys = AxisymmetricContactLayer(cfg=cfg).to(device)

    n_asp = cfg['physics']['n_asperities']
    R = cfg['physics']['radius']
    max_d = cfg['physics']['max_delta_ratio'] * R

    # 1. Create the "Ultimate Wall" (All Pseudo-Flat Punches touching at h=0)
    h_wall = torch.zeros(1, n_asp).to(device)
    n_wall = torch.ones(1, n_asp).to(device) * 3.0  # Max bounded exponent
    w_wall = torch.ones(1, n_asp).to(device) * 2.0 * R

    # Use the exact step count to ensure matching array sizes
    steps = cfg['data']['n_steps']
    ind = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)

    with torch.no_grad():
        # Unpack the 3 intensive outputs
        p_max, alpha_max, dp_dalpha_max = phys(h_wall, n_wall, w_wall, ind)

    return {
        "max_pressure": p_max.max().item(),
        "max_alpha": alpha_max.max().item(),
        "max_stiff": dp_dalpha_max.max().item() 
    }