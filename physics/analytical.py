import numpy as np
from scipy.special import gamma

"""
Load and contact area calculations for a collection of axisymmetric
asperities with power-law profiles.
Analytic solutions following Popov, 2019
"""

_datapoints_per_level = 100

def analytic(heights, ns, widths, E=1):
    '''For asperities with given heights, ns (exponents), and widths,
    following f(r) = height - r^n/(width^(n-1)), calculates
    load-area relationship: returns a tuple of (loads, areas)'''
    kappas = kappa(ns)
    endpoints = np.unique(heights)
    floor = 0
    
    # level = height - displacement, i.e. the z-position of the half-space
    levels = np.array([], dtype=np.float64)
    for level in endpoints:
        new_levels = np.linspace(floor, level, _datapoints_per_level)
        levels = np.hstack((levels, new_levels))
        floor = level

    loads = np.zeros_like(levels)
    areas = np.zeros_like(levels)
    for i, level in enumerate(levels):
        contact_radii = (np.maximum(heights - level, 0) /
                         kappas)**(1/ns) * widths**((ns-1)/ns)
        each_area = contact_radii**2 * np.pi
        each_load = E * 2 * ns / (ns + 1) * kappas * \
            widths**(1-ns) * contact_radii**(ns + 1)
        loads[i] = np.sum(each_load)
        areas[i] = np.sum(each_area)
    return loads, areas

def kappa(n):
    '''Scaling factor as defined in Popov, 2019'''
    return np.sqrt(np.pi) * gamma(n/2 + 1)/gamma((n+1)/2)