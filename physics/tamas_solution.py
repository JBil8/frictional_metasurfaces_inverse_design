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

def create_surface_grid(offsets, ns, widths, max_indentation):
    '''
    Generates surface grid dynamically. 
    Cell size expands to guarantee independent asperities (no elastic cross-talk).
    '''
    n_asp = len(offsets)
    grid_size = check_perfect_square(n_asp)
    
    # 1. Dynamically find the widest possible contact radius
    # Radius a = (delta * w^(n-1))^(1/n)
    safe_delta = max_indentation + 1e-6
    
    # Calculate max radius for each asperity based on its specific exponent
    contact_radii = (safe_delta * widths**(ns - 1))**(1/ns)
    max_r = np.max(contact_radii)
    
    if max_r <= 0: 
        max_r = np.max(widths) / 2.0 # Fallback

    # 2. Add massive padding (4x radius) to guarantee independent contacts
    cell_size = 4.0 * max_r
    L = cell_size * grid_size
    
    # 3. Keep resolution high for BEM accuracy (128 pixels per cell)
    N_elements = int(128 * grid_size)

    x = np.linspace(0, L, N_elements, endpoint=False)
    y = np.linspace(0, L, N_elements, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    surface = np.zeros_like(xx, dtype=np.float64) - 1e9

    # 4. Populate Grid
    for i in range(n_asp):
        h, n, w = offsets[i], ns[i], widths[i]
        
        row = i // grid_size
        col = i % grid_size
        
        cx = (col + 0.5) * cell_size
        cy = (row + 0.5) * cell_size
        
        z_offset = -h
        
        asp_topo = generate_asperity(xx, yy, cx, cy, z_offset, n, w).astype(np.float64)
        surface = np.maximum(surface, asp_topo)
        
    return L, N_elements, surface

def run_tamas_simulation(heights, ns, widths, target_loads, max_indentation, E_star=1.0, tol=1e-11):
    if tamaas is None:
        raise ImportError("Tamaas library not found.")

    L, N, surface_topo = create_surface_grid(heights, ns, widths, max_indentation)
    
    model = tamaas.Model(tamaas.model_type.basic_2d, [L, L], [N, N])
    solver = tamaas.PolonskyKeerRey(model, surface_topo, tol)
    
    areas = []
    print(f"  > Starting Tamaas BEM solve (Fixed Domain L={L:.4f})...")
    
    for load in target_loads:
        if load <= 0:
            areas.append(0.0)
            continue
            
        # Macroscopic pressure now uses a consistent, physically fixed L^2
        target_pressure = load / (L**2)
        
        try:
            solver.solve(target_pressure)
            contact_nodes = np.sum(model.traction > 0)
            pixel_area = (L / N) ** 2
            areas.append(contact_nodes * pixel_area)
        except Exception as e:
            print(f"    ! Solver failed at load {load:.4e}: {e}")
            areas.append(np.nan)
        
    return np.array(areas), surface_topo, L