import numpy as np

from numpy.polynomial.legendre import leggauss

from operators.fem import shape_functions


def transition_density(orbital_p : np.ndarray, orbital_r : np.ndarray) -> np.ndarray:
    psi_p = orbital_p.conj()
    psi_r = orbital_r

    return psi_p * psi_r


def integrate2d(f, rho_a, rho_b, z_a, z_b, order = 8):
    # nodes + weights
    points, weights = leggauss(order) 

    # nodes -> physical coordinates
    rho_points = 0.5 * (rho_b - rho_a) * points + 0.5 * (rho_b + rho_a)
    z_points = 0.5 * (z_b - z_a) * points + 0.5 * (z_b + z_a)
    
    # Jacobians
    J_rho = 0.5 * (rho_b - rho_a)
    J_z = 0.5 * (z_b - z_a)

    result = 0

    for i in range(order):
        for j in range(order):
            result += weights[i] * weights[j] * rho_points[i] * f(rho_points[i], z_points[j])

    result *= J_rho * J_z

    return result


def integrate(
    orbital_p : np.ndarray,
    orbital_q : np.ndarray,
    orbital_r : np.ndarray, 
    orbital_s : np.ndarray
) -> np.ndarray:
    pass


def calculate_g(orbitals):
    pass