import numpy as np

from numpy.polynomial.legendre import leggauss

from operators.fem import shape_functions


# ============================================================
# Utility functions
# ============================================================

def build_full_coefficients(coefficients, free_indices, n_total):
    """
    Restore a full coefficient vector.

    Degrees of freedom excluded by boundary conditions are zero.
    """

    coefficients = np.asarray(coefficients)
    free_indices = np.asarray(free_indices, dtype=int)

    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")

    if len(coefficients) != len(free_indices):
        raise ValueError("Incompatible coefficients and free_indices")

    full = np.zeros(n_total, dtype=coefficients.dtype)
    full[free_indices] = coefficients

    return full


def transition_density(orbital_p, orbital_r):
    """
    Calculate the transition density:

        f_pr(rho,z) = psi_p*(rho,z) * psi_r(rho,z)
    """

    orbital_p = np.asarray(orbital_p)
    orbital_r = np.asarray(orbital_r)

    if orbital_p.shape != orbital_r.shape:
        raise ValueError("Orbitals must have the same shape")

    return orbital_p.conj() * orbital_r


def _validate_mesh(rho_nodes, z_nodes):
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    if rho_nodes.ndim != 1 or z_nodes.ndim != 1:
        raise ValueError("Mesh coordinates must be one-dimensional")

    if len(rho_nodes) < 2 or len(z_nodes) < 2:
        raise ValueError("Each mesh must contain at least two nodes")

    if np.any(np.diff(rho_nodes) <= 0):
        raise ValueError("rho_nodes must be strictly increasing")

    if np.any(np.diff(z_nodes) <= 0):
        raise ValueError("z_nodes must be strictly increasing")

    if rho_nodes[0] < 0:
        raise ValueError("Radial coordinates cannot be negative")

    return rho_nodes, z_nodes


def _element_nodes(i, j, n_z):
    """
    Return global indices of the four Q1 element nodes.

    Local order:
        0: (i,   j)
        1: (i+1, j)
        2: (i,   j+1)
        3: (i+1, j+1)
    """

    return np.array([
        i * n_z + j,
        (i + 1) * n_z + j,
        i * n_z + (j + 1),
        (i + 1) * n_z + (j + 1),
    ])


# ============================================================
# Angular Coulomb kernel
# ============================================================

def angular_coulomb_kernel(
    rho1,
    z1,
    rho2,
    z2,
    k,
    quadrature_order=32,
):
    r"""
    Calculate the angular Coulomb kernel:

        K_k = 1/(2*pi) * integral_0^(2*pi)
              exp(i*k*theta) / distance(theta) dtheta

    where

        distance(theta)^2 =
            rho1^2 + rho2^2
            - 2*rho1*rho2*cos(theta)
            + (z1-z2)^2.
    """

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")

    points, weights = leggauss(quadrature_order)

    theta = np.pi * (points + 1.0)
    theta_weights = np.pi * weights

    distance_squared = (
        rho1**2
        + rho2**2
        - 2.0 * rho1 * rho2 * np.cos(theta)
        + (z1 - z2)**2
    )

    distances = np.sqrt(np.maximum(distance_squared, 0.0))

    if np.any(distances == 0.0):
        raise FloatingPointError(
            "Angular kernel is singular at coincident coordinates"
        )

    integrand = np.exp(1j * k * theta) / distances

    result = np.sum(theta_weights * integrand) / (2.0 * np.pi)

    # The exact kernel is real because the angular integrand
    # has a symmetric real denominator.
    return float(np.real(result))


# ============================================================
# Element quadrature
# ============================================================

def _build_element_quadrature(
    density,
    rho_nodes,
    z_nodes,
    quadrature_order,
):
    """
    Precompute quadrature points for every FEM element.

    Each point stores:
        rho,
        z,
        weight * rho * interpolated_density.
    """

    n_rho = len(rho_nodes)
    n_z = len(z_nodes)

    points, weights = leggauss(quadrature_order)

    density_flat = density.ravel()

    elements = []

    for i in range(n_rho - 1):

        rho_a = rho_nodes[i]
        rho_b = rho_nodes[i + 1]

        J_rho = 0.5 * (rho_b - rho_a)

        for j in range(n_z - 1):

            z_a = z_nodes[j]
            z_b = z_nodes[j + 1]

            J_z = 0.5 * (z_b - z_a)

            nodes = _element_nodes(i, j, n_z)
            local_density = density_flat[nodes]

            element_points = []

            for a, xi in enumerate(points):

                rho = (
                    0.5 * (rho_a + rho_b)
                    + J_rho * xi
                )

                for b, eta in enumerate(points):

                    z = (
                        0.5 * (z_a + z_b)
                        + J_z * eta
                    )

                    N, _, _ = shape_functions(xi, eta)

                    interpolated_density = np.dot(
                        N,
                        local_density,
                    )

                    weight = (
                        weights[a]
                        * weights[b]
                        * J_rho
                        * J_z
                        * rho
                    )

                    element_points.append((
                        rho,
                        z,
                        weight * interpolated_density,
                    ))

            elements.append(element_points)

    return elements


# ============================================================
# Four-dimensional integration
# ============================================================

def integrate_transition_densities(
    f_pr,
    f_qs,
    rho_nodes,
    z_nodes,
    k,
    quadrature_order=4,
    angular_quadrature_order=32,
):
    r"""
    Calculate:

        integral rho1 drho1 dz1 rho2 drho2 dz2
            f_pr(rho1,z1)
            f_qs(rho2,z2)
            K_k(rho1,z1,rho2,z2)

    Transition densities are supplied as nodal arrays.
    Q1 interpolation is used inside each FEM element.
    """

    f_pr = np.asarray(f_pr)
    f_qs = np.asarray(f_qs)

    rho_nodes, z_nodes = _validate_mesh(
        rho_nodes,
        z_nodes,
    )

    expected_shape = (len(rho_nodes), len(z_nodes))

    if f_pr.shape != expected_shape:
        raise ValueError(
            f"f_pr must have shape {expected_shape}, "
            f"got {f_pr.shape}"
        )

    if f_qs.shape != expected_shape:
        raise ValueError(
            f"f_qs must have shape {expected_shape}, "
            f"got {f_qs.shape}"
        )

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")

    if angular_quadrature_order < 1:
        raise ValueError(
            "angular_quadrature_order must be positive"
        )

    # Precompute all one-electron quadrature points.
    elements_pr = _build_element_quadrature(
        f_pr,
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    elements_qs = _build_element_quadrature(
        f_qs,
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    result = 0.0 + 0.0j

    # Sum over all pairs of FEM elements.
    for element_pr in elements_pr:

        for element_qs in elements_qs:

            # Sum over quadrature points of both electrons.
            for rho1, z1, weight1 in element_pr:

                for rho2, z2, weight2 in element_qs:

                    kernel = angular_coulomb_kernel(
                        rho1,
                        z1,
                        rho2,
                        z2,
                        k,
                        angular_quadrature_order,
                    )

                    result += weight1 * weight2 * kernel

    return result


# ============================================================
# One two-electron integral
# ============================================================

def integrate(
    orbital_p,
    orbital_q,
    orbital_r,
    orbital_s,
    rho_nodes,
    z_nodes,
    m_p,
    m_q,
    m_r,
    m_s,
    quadrature_order=4,
    angular_quadrature_order=32,
):
    r"""
    Calculate:

        g_pqrs = <pq | 1/r12 | rs>

    with the convention:

        g_pqrs =
            integral phi_p*(1) phi_q*(2)
                    1/r12
                    phi_r(1) phi_s(2) d1 d2

    The angular selection rule is:

        m_p + m_q = m_r + m_s.
    """

    magnetic_numbers = (m_p, m_q, m_r, m_s)

    if not all(
        isinstance(m, (int, np.integer))
        for m in magnetic_numbers
    ):
        raise TypeError("All magnetic quantum numbers must be integers")

    if m_p + m_q != m_r + m_s:
        return 0.0

    f_pr = transition_density(
        orbital_p,
        orbital_r,
    )

    f_qs = transition_density(
        orbital_q,
        orbital_s,
    )

    k = m_p - m_r

    return integrate_transition_densities(
        f_pr=f_pr,
        f_qs=f_qs,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        k=k,
        quadrature_order=quadrature_order,
        angular_quadrature_order=angular_quadrature_order,
    )


# ============================================================
# Full two-electron integral tensor
# ============================================================

def calculate_g(
    orbital_sets,
    rho_nodes,
    z_nodes,
    quadrature_order=4,
    angular_quadrature_order=32,
):
    r"""
    Calculate the complete spatial-orbital tensor g[p,q,r,s].

    Parameters
    ----------
    orbital_sets : iterable of OrbitalSet
        Each OrbitalSet may contain several orbitals.
        All orbitals from all supplied sets are included.

    rho_nodes, z_nodes : arrays
        Physical FEM mesh coordinates.

    Returns
    -------
    g : ndarray, shape (n,n,n,n)
        Two-electron integral tensor.

    orbital_m : ndarray, shape (n,)
        Magnetic quantum number of each orbital.

    orbital_energies : ndarray, shape (n,)
        One-electron energy of each orbital.

    Notes
    -----
    The ordering of orbitals is:
        first OrbitalSet, all its orbitals;
        second OrbitalSet, all its orbitals;
        etc.
    """

    from orbitals.evaluate import evaluate_orbital_grid

    rho_nodes, z_nodes = _validate_mesh(
        rho_nodes,
        z_nodes,
    )

    # Flatten all orbital sets into one spatial-orbital list.
    orbital_grids = []
    orbital_m = []
    orbital_energies = []

    for orbital_set in orbital_sets:

        grids = evaluate_orbital_grid(
            orbital_set,
            rho_nodes,
            z_nodes,
            orbital_index=0,
        )

        # Evaluate all orbitals in this set.
        for orbital_index in range(orbital_set.n_orbitals):

            if orbital_index == 0:
                orbital_grid = grids
            else:
                orbital_grid = evaluate_orbital_grid(
                    orbital_set,
                    rho_nodes,
                    z_nodes,
                    orbital_index=orbital_index,
                )

            orbital_grids.append(orbital_grid)
            orbital_m.append(orbital_set.m)
            orbital_energies.append(
                orbital_set.energies[orbital_index]
            )

    n_orbitals = len(orbital_grids)

    if n_orbitals == 0:
        raise ValueError("No orbitals were supplied")

    orbital_m = np.asarray(orbital_m, dtype=int)
    orbital_energies = np.asarray(orbital_energies)

    # Cache transition densities f_pr.
    transition_densities = {}

    for p in range(n_orbitals):
        for r in range(n_orbitals):

            transition_densities[p, r] = transition_density(
                orbital_grids[p],
                orbital_grids[r],
            )

    # Allocate the full four-index tensor.
    g = np.zeros(
        (n_orbitals,) * 4,
        dtype=complex,
    )

    # Calculate only symmetry-allowed integrals.
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):

                    if (
                        orbital_m[p] + orbital_m[q]
                        != orbital_m[r] + orbital_m[s]
                    ):
                        continue

                    k = int(orbital_m[p] - orbital_m[r])

                    g[p, q, r, s] = integrate_transition_densities(
                        f_pr=transition_densities[p, r],
                        f_qs=transition_densities[q, s],
                        rho_nodes=rho_nodes,
                        z_nodes=z_nodes,
                        k=k,
                        quadrature_order=quadrature_order,
                        angular_quadrature_order=angular_quadrature_order,
                    )

    return g, orbital_m, orbital_energies