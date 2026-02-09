import numpy as np
import math
from mpi4py import MPI

# Try-except block allows the code to be imported even if tamaas isn't installed
try:
    import tamaas
except ImportError:
    tamaas = None

def check_perfect_square(n):
    root = int(math.isqrt(n))
    if root * root != n:
        raise ValueError(f"Number of asperities ({n}) must be a perfect square (e.g., 4, 9, 16, 25) for grid generation.")
    return root

def generate_asperity(xx, yy, center_x, center_y, z_offset, n, width):
    '''
    Generates a single axisymmetric asperity.
    z = z_offset - r^n / width^(n-1)
    '''
    r_sq = (xx - center_x)**2 + (yy - center_y)**2
    # Avoid negative bases for fractional powers: (r^2)^(n/2)
    shape = -np.power(r_sq, n / 2.0) / (width**(n - 1))
    return shape + z_offset

def create_surface_grid(offsets, ns, widths, max_indentation):
    '''
    Generates surface grid.
    Args:
        offsets: Array of Gap Offsets (h). h=0 means tallest asperity.
        max_indentation: The physical max depth to size the grid correctly.
    '''
    n_asp = len(offsets)
    grid_size = check_perfect_square(n_asp)
    
    # 1. Determine Grid Dimensions based on MAX INDENTATION
    # The max contact radius happens when indentation is deepest.
    # Radius a = (delta * w^(n-1))^(1/n)
    # We use max_indentation for delta to be safe.
    # We add a small epsilon to delta to avoid zero radius.
    safe_delta = max_indentation + 1e-6
    contact_radii = (safe_delta * widths**(ns - 1))**(1/ns)
    max_r = np.max(contact_radii)
    
    if max_r == 0: max_r = 1.0

    # Add 50% padding to prevent interaction
    cell_size = 4.0 * max_r
    L = cell_size * grid_size
    
    # 2. Resolution (BEM needs good resolution)
    # 128 pixels per cell is a safe high-fidelity standard
    N_elements = int(128 * 2 * grid_size)

    x = np.linspace(0, L, N_elements, endpoint=False)
    y = np.linspace(0, L, N_elements, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # CHANGE HERE: Force float64
    surface = np.zeros_like(xx, dtype=np.float64) - 1e9

    # 3. Populate Grid
    for i in range(n_asp):
        h, n, w = offsets[i], ns[i], widths[i]
        
        row = i // grid_size
        col = i % grid_size
        
        cx = (col + 0.5) * cell_size
        cy = (row + 0.5) * cell_size
        
        z_offset = -h
        
        # Ensure topography is also float64
        asp_topo = generate_asperity(xx, yy, cx, cy, z_offset, n, w).astype(np.float64)
        
        surface = np.maximum(surface, asp_topo)
        
    return L, N_elements, surface # Now guaranteed to be float64

def run_tamas_simulation(heights, ns, widths, target_loads, max_indentation, E_star=1.0, tol=1e-11):
    '''
    Args:
        heights: These are actually OFFSETS (h) from the NN.
        max_indentation: Required to size the domain non-zero.
    '''
    if tamaas is None:
        raise ImportError("Tamaas library not found.")

    # 1. Generate Surface (Pass max_indentation!)
    L, N, surface_topo = create_surface_grid(heights, ns, widths, max_indentation)
    
    if L == 0:
        raise ValueError("Domain Size L is zero. Check max_indentation or widths.")

    # 2. Setup Model
    model = tamaas.Model(tamaas.model_type.basic_2d, [L, L], [N, N])
    
    # Solve contact.
    # We solve for a sequence of target loads.
    solver = tamaas.PolonskyKeerRey(model, surface_topo, tol)
    
    areas = []
    print(f"  > Starting Tamaas BEM solve (Domain L={L:.4f})...")
    
    for load in target_loads:
        if load <= 0:
            areas.append(0.0)
            continue
            
        # Pressure = Force / Area
        target_pressure = load / (L**2)
        
        try:
            solver.solve(target_pressure)
            
            # Calculate Area
            contact_nodes = np.sum(model.traction > 0)
            pixel_area = (L / N) ** 2
            total_area = contact_nodes * pixel_area
            areas.append(total_area)
        except Exception as e:
            print(f"    ! Solver failed at load {load:.4e}: {e}")
            areas.append(np.nan)
        
    return np.array(areas), surface_topo, L