# utils/normalization.py
import torch
from physics.differentiable import AxisymmetricContactLayer

def get_theoretical_limits(cfg, device):
    """
    Calculates the ABSOLUTE maximum Load, Area, and Stiffness 
    possible for the given physics configuration (16 Flat Punches).
    """
    phys = AxisymmetricContactLayer(E_star=cfg['physics']['E_star']).to(device)
    
    n_asp = cfg['physics']['n_asperities']
    R = cfg['physics']['radius']
    max_d = cfg['physics']['max_delta_ratio'] * R
    
    # 1. Create the "Ultimate Wall" (All Flat Punches touching)
    h_wall = torch.zeros(1, n_asp).to(device)
    n_wall = torch.ones(1, n_asp).to(device) * 8.0 # Max Exponent
    w_wall = torch.ones(1, n_asp).to(device) * 2.0 * R
    
    # 2. Indent fully
    # We use a high resolution to capture max stiffness accurately
    steps = 100
    ind = torch.linspace(0, max_d, steps).unsqueeze(0).to(device)
    
    with torch.no_grad():
        l_max, a_max = phys(h_wall, n_wall, w_wall, ind)
    
    # 3. Calculate Max Stiffness
    # Prepend 0 to keep shape
    stiff = torch.diff(l_max, dim=1, prepend=torch.zeros(1, 1).to(device))
    
    return {
        "max_load": l_max.max().item(),
        "max_area": a_max.max().item(),
        "max_stiff": stiff.max().item()
    }