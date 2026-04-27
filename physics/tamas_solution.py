import numpy as np
import math

try:
    import tamaas
except ImportError:
    tamaas = None

def check_perfect_square(n):
    root = int(math.isqrt(n))
    if root * root != n:
        raise ValueError(f"Number of asperities ({n}) must be a perfect square.")
    return root

def generate_asperity(xx, yy, center_x, center_y, z_offset, n, width):
    r_sq = (xx - center_x)**2 + (yy - center_y)**2
    shape = -np.power(r_sq, n / 2.0) / (width**(n - 1))
    return shape + z_offset

def create_surface_grid(offsets, ns, widths, L):
    '''
    Generates surface grid using the GLOBALLY FIXED domain size L.
    '''
    n_asp = len(offsets)
    grid_size = check_perfect_square(n_asp)
    
    # Cell size is now strictly dictated by the global L
    cell_size = L / grid_size
    
    # Keep resolution high for BEM accuracy (128 pixels per cell)
    N_elements = int(128 * grid_size)

    x = np.linspace(0, L, N_elements, endpoint=False)
    y = np.linspace(0, L, N_elements, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    surface = np.zeros_like(xx, dtype=np.float64) - 1e9

    # Populate Grid
    for i in range(n_asp):
        h, n, w = offsets[i], ns[i], widths[i]
        
        row = i // grid_size
        col = i % grid_size
        
        cx = (col + 0.5) * cell_size
        cy = (row + 0.5) * cell_size
        
        z_offset = -h
        
        asp_topo = generate_asperity(xx, yy, cx, cy, z_offset, n, w).astype(np.float64)
        surface = np.maximum(surface, asp_topo)
        
    return N_elements, surface

def run_tamas_simulation(heights, ns, widths, target_pressures, L, E_star=1.0, tol=1e-11):
    if tamaas is None:
        raise ImportError("Tamaas library not found.")

    N, surface_topo = create_surface_grid(heights, ns, widths, L)
    
    model = tamaas.Model(tamaas.model_type.basic_2d, [L, L], [N, N])
    solver = tamaas.PolonskyKeerRey(model, surface_topo, tol)
    
    alphas = []
    final_pressure_field = None  
    print(f"  > Starting Tamaas BEM solve (Fixed Domain L={L:.4f})...")
    
    for pressure in target_pressures:
        if pressure <= 0:
            alphas.append(0.0)
            continue
            
        try:
            solver.solve(pressure)
            
            contact_nodes = np.sum(model.traction > 0)
            total_nodes = N * N
            contact_fraction = contact_nodes / total_nodes
            alphas.append(contact_fraction)
            
            final_pressure_field = np.copy(model.traction) 
            
        except Exception as e:
            print(f"    ! Solver failed at pressure {pressure:.4e}: {e}")
            alphas.append(np.nan)
        
    return np.array(alphas), surface_topo, final_pressure_field, L