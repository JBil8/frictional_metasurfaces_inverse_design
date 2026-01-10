import numpy as np
import tamaas

_datapoints_per_level = 100

def experimental(heights, ns, widths, loads, tol=1e-11):
    '''For asperities with given heights, ns (exponents), and widths,
    following f(r) = height - r^n/(width^(n-1)), calculates contact areas
    at each load in loads'''
    L, N, surface = experimental_surface(heights, ns, widths)

    model = tamaas.Model(tamaas.model_type.basic_2d, [L, L], [N, N])
    solver = tamaas.PolonskyKeerRey(model, surface, tol)

    areas = np.zeros_like(loads)
    for i, load in enumerate(loads):
        if load == 0:
            continue
        solver.solve(load / L ** 2)
        areas[i] = np.sum(model.traction != 0) / N**2 * L * L

    return areas

def generate_asperity(xx, yy, center_x, center_y, height, n, width):
    '''Given a meshgrid of (xx, yy), generate an asperity centered
    at (center_x, center_y) with specified height, n (exponent), and width
    following f(r) = height - r^n/(width^(n-1))'''
    z = -((xx-center_x)**2 + (yy-center_y)**2)**(n / 2) / width**(n - 1)
    z += height
    return np.maximum(z, 0)


def experimental_surface(heights, ns, widths):
    '''For asperities with given heights, ns (exponents), and widths,
    following f(r) = height - r^n/(width^(n-1)), generates a surface
    for use with Tamaas. Returns (L, N, surface), where L is the 
    side length and N is the number of elements per row/column.'''
    # TODO: Figure out padding between cells
    max_rad = np.max(heights**(1/ns) * widths**((ns-1)/ns))*1.1
    side_length = np.ceil(np.sqrt(len(heights)))
    L = 2 * max_rad * side_length
    # TODO: Figure out mesh size scaling
    N = int(np.ceil(512 * side_length**(1/5)))

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    surface = np.zeros_like(xx)

    for i, (h, n, w) in enumerate(zip(heights, ns, widths)):
        x = (2 * (i % side_length) + 1) * max_rad
        y = (2 * (i // side_length) + 1) * max_rad
        surface = np.maximum(surface, generate_asperity(xx, yy, x, y, h, n, w))
    return L, N, surface
