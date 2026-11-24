import numpy as np

from numpy.polynomial.legendre import leggauss
from operators.fem import shape_functions

from scipy.sparse import coo_matrix

def assemble_local_poisson_matrix(
    rho_a, rho_b,
    z_a, z_b,
    k,
    quadrature_order=8
):
    points, weights = leggauss(quadrature_order)

    J_rho = 0.5 * (rho_b - rho_a)
    J_z = 0.5 * (z_b - z_a)

    K_local = np.zeros((4, 4), dtype=float)

    for i in range(quadrature_order):
        for j in range(quadrature_order):

            xi = points[i]
            eta = points[j]

            rho = J_rho * xi + 0.5 * (rho_a + rho_b)
            z = J_z * eta + 0.5 * (z_a + z_b)

            N, dN_dxi, dN_deta = shape_functions(xi, eta)

            dN_drho = dN_dxi / J_rho
            dN_dz = dN_deta / J_z

            weight = weights[i] * weights[j] * J_rho * J_z

            K_local += weight * (
                rho * (
                    np.outer(dN_drho, dN_drho)
                    + np.outer(dN_dz, dN_dz)
                )
                + (k**2 / rho) * np.outer(N, N)
            )

    return K_local


def assemble_poisson_matrix(
    rho_nodes,
    z_nodes,
    k,
    quadrature_order=8,
    return_full=False,
):
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    nr = len(rho_nodes)
    nz = len(z_nodes)
    n_total = nr * nz

    rows = []
    cols = []
    data = []

    for i in range(nr - 1):
        for j in range(nz - 1):

            K_local = assemble_local_poisson_matrix(
                rho_a=rho_nodes[i],
                rho_b=rho_nodes[i + 1],
                z_a=z_nodes[j],
                z_b=z_nodes[j + 1],
                k=k,
                quadrature_order=quadrature_order
            )

            nodes = np.array([
                i * nz + j,
                (i + 1) * nz + j,
                i * nz + (j + 1),
                (i + 1) * nz + (j + 1),
            ])

            row, col = np.meshgrid(nodes, nodes, indexing="ij")

            rows.extend(row.ravel())
            cols.extend(col.ravel())
            data.extend(K_local.ravel())

    K_full = coo_matrix(
        (data, (rows, cols)),
        shape=(n_total, n_total)
    ).tocsr()

    # Граничные условия Дирихле
    free_mask = np.ones(n_total, dtype=bool)

    # Внешняя радиальная граница
    free_mask[(nr - 1) * nz:nr * nz] = False

    # Нижняя и верхняя границы по z
    free_mask[np.arange(nr) * nz] = False
    free_mask[np.arange(nr) * nz + nz - 1] = False

    # Ось rho = 0 при k != 0
    if k != 0 and np.isclose(rho_nodes[0], 0.0):
        free_mask[:nz] = False

    free_indices = np.flatnonzero(free_mask)

    K = K_full[free_indices, :][:, free_indices].tocsc()

    if return_full:
        return K, free_indices, K_full.tocsr()

    return K, free_indices

def assemble_poisson_rhs(
    density,
    rho_nodes,
    z_nodes,
    free_indices,
    quadrature_order=8,
):
    r"""
    Assemble the right-hand side of the cylindrical Poisson equation

        -[1/rho d/drho(rho du/drho) + d2u/dz2
          - k^2/rho^2 u] = 2 density.

    Its weak-form load vector is

        b_mu = 2 * integral rho * N_mu * density d(rho) dz.

    Parameters
    ----------
    density : ndarray, shape (n_rho, n_z)
        Nodal values of the source density. Complex values are supported.
    rho_nodes, z_nodes : 1D array_like
        Physical mesh coordinates.
    free_indices : 1D array_like of int
        The free global DOF indices returned by assemble_poisson_matrix
        for the same mesh and k value.
    quadrature_order : int
        Gauss-Legendre order in each reference-element direction.

    Returns
    -------
    ndarray
        Right-hand-side vector restricted to free degrees of freedom.
    """

    density = np.asarray(density)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    free_indices = np.asarray(free_indices, dtype=int)

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

    expected_shape = (len(rho_nodes), len(z_nodes))
    if density.shape != expected_shape:
        raise ValueError(
            f"density must have shape {expected_shape}, got {density.shape}"
        )

    if not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")

    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")

    n_rho, n_z = expected_shape
    n_total = n_rho * n_z

    if free_indices.ndim != 1:
        raise ValueError("free_indices must be one-dimensional")

    if np.any(free_indices < 0) or np.any(free_indices >= n_total):
        raise ValueError("free_indices contain out-of-range indices")

    if len(np.unique(free_indices)) != len(free_indices):
        raise ValueError("free_indices must not contain duplicates")

    points, weights = leggauss(quadrature_order)

    # Keep complex transition densities complex; real densities remain real.
    dtype = np.result_type(density.dtype, np.float64)
    b_full = np.zeros(n_total, dtype=dtype)
    density_flat = density.ravel()

    for i in range(n_rho - 1):
        rho_a = rho_nodes[i]
        rho_b = rho_nodes[i + 1]
        J_rho = 0.5 * (rho_b - rho_a)

        for j in range(n_z - 1):
            z_a = z_nodes[j]
            z_b = z_nodes[j + 1]
            J_z = 0.5 * (z_b - z_a)

            nodes = np.array([
                i * n_z + j,
                (i + 1) * n_z + j,
                i * n_z + (j + 1),
                (i + 1) * n_z + (j + 1),
            ])
            local_density = density_flat[nodes]
            local_rhs = np.zeros(4, dtype=dtype)

            for xi, wx in zip(points, weights):
                rho = 0.5 * (rho_a + rho_b) + J_rho * xi

                for eta, wy in zip(points, weights):
                    N, _, _ = shape_functions(xi, eta)
                    z = 0.5 * (z_a + z_b) + J_z * eta

                    density_at_point = np.dot(N, local_density)
                    weight = wx * wy * J_rho * J_z

                    local_rhs += (
                        2.0 * weight * rho * N * density_at_point
                    )

            # Assemble the local vector into the global vector.
            np.add.at(b_full, nodes, local_rhs)

    return b_full[free_indices]




def _transition_charge(
    orbital_q,
    orbital_s,
    rho_nodes,
    z_nodes,
    quadrature_order,
):
    """Integrate rho * conj(psi_q) * psi_s over the meridional domain."""
    points, weights = leggauss(quadrature_order)
    n_rho, n_z = orbital_q.shape
    q_flat = orbital_q.ravel()
    s_flat = orbital_s.ravel()
    charge = 0.0 + 0.0j

    for i in range(n_rho - 1):
        rho_a, rho_b = rho_nodes[i:i + 2]
        J_rho = 0.5 * (rho_b - rho_a)
        for j in range(n_z - 1):
            z_a, z_b = z_nodes[j:j + 2]
            J_z = 0.5 * (z_b - z_a)
            nodes = np.array([
                i * n_z + j,
                (i + 1) * n_z + j,
                i * n_z + j + 1,
                (i + 1) * n_z + j + 1,
            ])
            q_local = q_flat[nodes]
            s_local = s_flat[nodes]
            for xi, wx in zip(points, weights):
                rho = 0.5 * (rho_a + rho_b) + J_rho * xi
                for eta, wy in zip(points, weights):
                    N, _, _ = shape_functions(xi, eta)
                    q_value = np.dot(N, q_local)
                    s_value = np.dot(N, s_local)
                    charge += (
                        wx * wy * J_rho * J_z * rho
                        * np.conj(q_value) * s_value
                    )
    return charge


def _monopole_boundary_values(rho_nodes, z_nodes, charge):
    """Approximate the k=0 far-field boundary by Q/r."""
    rho_grid = np.asarray(rho_nodes)[:, None]
    z_grid = np.asarray(z_nodes)[None, :]
    radius = np.sqrt(rho_grid**2 + z_grid**2)
    values = np.zeros(
        (len(rho_nodes), len(z_nodes)),
        dtype=np.result_type(charge, np.float64),
    )
    boundary = np.zeros(values.shape, dtype=bool)
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    values[boundary] = charge / radius[boundary]
    return values


def solve_poisson(
    density,
    rho_nodes,
    z_nodes,
    k,
    quadrature_order=8,
):
    r"""Solve the cylindrical Poisson equation for one angular sector.

    The equation is

        -[1/rho d/drho(rho du/drho) + d2u/dz2
          - k**2/rho**2 * u] = 2 * density.

    Dirichlet boundary values are zero. For k == 0 the symmetry axis
    is retained as a natural boundary; for k != 0 the axis is fixed
    to zero for regularity.
    """
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    density = np.asarray(density)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    expected_shape = (len(rho_nodes), len(z_nodes))
    if density.shape != expected_shape:
        raise ValueError(
            f"density must have shape {expected_shape}, got {density.shape}"
        )

    K, free_indices = assemble_poisson_matrix(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        k=k,
        quadrature_order=quadrature_order,
    )

    b = assemble_poisson_rhs(
        density=density,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        free_indices=free_indices,
        quadrature_order=quadrature_order,
    )

    from scipy.sparse.linalg import spsolve
    coefficients = spsolve(K, b)

    potential = np.zeros(
        expected_shape,
        dtype=np.result_type(coefficients.dtype, density.dtype, np.float64),
    )
    potential.ravel()[free_indices] = coefficients

    return potential


def integrate_poisson_potential(
    density,
    potential,
    rho_nodes,
    z_nodes,
    quadrature_order=8,
):
    r"""Evaluate integral rho * density * potential d(rho) dz.

    Both fields are represented by nodal Q1 values on the same
    physical (rho, z) mesh.
    """
    density = np.asarray(density)
    potential = np.asarray(potential)
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

    expected_shape = (len(rho_nodes), len(z_nodes))
    if density.shape != expected_shape:
        raise ValueError(
            f"density must have shape {expected_shape}, got {density.shape}"
        )
    if potential.shape != expected_shape:
        raise ValueError(
            f"potential must have shape {expected_shape}, got {potential.shape}"
        )
    if not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")

    points, weights = leggauss(quadrature_order)
    n_rho, n_z = expected_shape
    density_flat = density.ravel()
    potential_flat = potential.ravel()

    result = np.zeros(
        (),
        dtype=np.result_type(density.dtype, potential.dtype, np.float64),
    ).item()

    for i in range(n_rho - 1):
        rho_a, rho_b = rho_nodes[i:i + 2]
        J_rho = 0.5 * (rho_b - rho_a)

        for j in range(n_z - 1):
            z_a, z_b = z_nodes[j:j + 2]
            J_z = 0.5 * (z_b - z_a)

            nodes = np.array([
                i * n_z + j,
                (i + 1) * n_z + j,
                i * n_z + (j + 1),
                (i + 1) * n_z + (j + 1),
            ])
            local_density = density_flat[nodes]
            local_potential = potential_flat[nodes]

            for xi, wx in zip(points, weights):
                rho = 0.5 * (rho_a + rho_b) + J_rho * xi

                for eta, wy in zip(points, weights):
                    N, _, _ = shape_functions(xi, eta)
                    density_value = np.dot(N, local_density)
                    potential_value = np.dot(N, local_potential)
                    result += (
                        wx * wy * J_rho * J_z
                        * rho * density_value * potential_value
                    )

    return result




def assemble_poisson_rhs_from_orbitals(
    orbital_q,
    orbital_s,
    rho_nodes,
    z_nodes,
    free_indices,
    quadrature_order=8,
):
    """Assemble the Poisson load using the product of Q1 orbitals.

    Unlike interpolating nodal values of psi_q.conj()*psi_s, this
    evaluates each Q1 orbital at every quadrature point and forms
    their product there.
    """
    orbital_q = np.asarray(orbital_q)
    orbital_s = np.asarray(orbital_s)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    free_indices = np.asarray(free_indices, dtype=int)

    expected = (len(rho_nodes), len(z_nodes))
    if orbital_q.shape != expected or orbital_s.shape != expected:
        raise ValueError(f"orbital grids must have shape {expected}")
    if not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")

    n_rho, n_z = expected
    n_total = n_rho * n_z
    if free_indices.ndim != 1 or np.any(free_indices < 0) or np.any(free_indices >= n_total):
        raise ValueError("Invalid free_indices")

    dtype = np.result_type(orbital_q.dtype, orbital_s.dtype, np.float64)
    b_full = np.zeros(n_total, dtype=dtype)
    points, weights = leggauss(quadrature_order)

    q_flat = orbital_q.ravel()
    s_flat = orbital_s.ravel()

    for i in range(n_rho - 1):
        rho_a, rho_b = rho_nodes[i:i + 2]
        J_rho = 0.5 * (rho_b - rho_a)

        for j in range(n_z - 1):
            z_a, z_b = z_nodes[j:j + 2]
            J_z = 0.5 * (z_b - z_a)
            nodes = np.array([
                i * n_z + j,
                (i + 1) * n_z + j,
                i * n_z + j + 1,
                (i + 1) * n_z + j + 1,
            ])
            q_local = q_flat[nodes]
            s_local = s_flat[nodes]
            local_rhs = np.zeros(4, dtype=dtype)

            for xi, wx in zip(points, weights):
                for eta, wy in zip(points, weights):
                    N, _, _ = shape_functions(xi, eta)
                    rho = 0.5 * (rho_a + rho_b) + J_rho * xi
                    q_value = np.dot(N, q_local)
                    s_value = np.dot(N, s_local)
                    source = np.conj(q_value) * s_value
                    local_rhs += (
                        2.0 * wx * wy * J_rho * J_z
                        * rho * N * source
                    )

            np.add.at(b_full, nodes, local_rhs)

    return b_full[free_indices]


def integrate_poisson_orbitals(
    orbital_p,
    orbital_q,
    orbital_r,
    orbital_s,
    rho_nodes,
    z_nodes,
    k,
    quadrature_order=8,
):
    r"""Calculate g_pqrs by the Poisson method from four Q1 orbital grids.

    The transition densities are formed from the interpolated orbitals
    at quadrature points, rather than interpolating their nodal products.
    """
    orbital_p = np.asarray(orbital_p)
    orbital_q = np.asarray(orbital_q)
    orbital_r = np.asarray(orbital_r)
    orbital_s = np.asarray(orbital_s)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    expected = (len(rho_nodes), len(z_nodes))
    for name, orbital in (
        ("orbital_p", orbital_p),
        ("orbital_q", orbital_q),
        ("orbital_r", orbital_r),
        ("orbital_s", orbital_s),
    ):
        if orbital.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")

    K, free_indices, K_full = assemble_poisson_matrix(
        rho_nodes, z_nodes, int(k), quadrature_order,
        return_full=True,
    )
    rhs = assemble_poisson_rhs_from_orbitals(
        orbital_q, orbital_s,
        rho_nodes, z_nodes, free_indices,
        quadrature_order,
    )

    boundary_mask = np.ones(expected, dtype=bool)
    boundary_mask[-1, :] = False
    boundary_mask[:, 0] = False
    boundary_mask[:, -1] = False
    if k != 0 and np.isclose(rho_nodes[0], 0.0):
        boundary_mask[0, :] = False

    boundary_values = np.zeros(
        expected,
        dtype=np.result_type(orbital_q.dtype, orbital_s.dtype, np.float64),
    )
    if k == 0:
        charge = _transition_charge(
            orbital_q, orbital_s, rho_nodes, z_nodes, quadrature_order
        )
        boundary_values = _monopole_boundary_values(
            rho_nodes, z_nodes, charge
        )

    boundary_indices = np.flatnonzero(~boundary_mask.ravel())
    boundary_vector = boundary_values.ravel()[boundary_indices]
    rhs = rhs - K_full[free_indices, :][:, boundary_indices] @ boundary_vector

    from scipy.sparse.linalg import spsolve
    coefficients = spsolve(K, rhs)

    potential = boundary_values.copy()
    potential.ravel()[free_indices] = coefficients

    points, weights = leggauss(quadrature_order)
    n_rho, n_z = expected
    p_flat = orbital_p.ravel()
    r_flat = orbital_r.ravel()
    u_flat = potential.ravel()

    result = np.zeros(
        (),
        dtype=np.result_type(
            orbital_p.dtype, orbital_r.dtype, potential.dtype, np.float64
        ),
    ).item()

    for i in range(n_rho - 1):
        rho_a, rho_b = rho_nodes[i:i + 2]
        J_rho = 0.5 * (rho_b - rho_a)

        for j in range(n_z - 1):
            z_a, z_b = z_nodes[j:j + 2]
            J_z = 0.5 * (z_b - z_a)
            nodes = np.array([
                i * n_z + j,
                (i + 1) * n_z + j,
                i * n_z + j + 1,
                (i + 1) * n_z + j + 1,
            ])
            p_local = p_flat[nodes]
            r_local = r_flat[nodes]
            u_local = u_flat[nodes]

            for xi, wx in zip(points, weights):
                for eta, wy in zip(points, weights):
                    N, _, _ = shape_functions(xi, eta)
                    rho = 0.5 * (rho_a + rho_b) + J_rho * xi
                    p_value = np.dot(N, p_local)
                    r_value = np.dot(N, r_local)
                    u_value = np.dot(N, u_local)
                    density_pr = np.conj(p_value) * r_value
                    result += (
                        wx * wy * J_rho * J_z
                        * rho * density_pr * u_value
                    )

    return result
