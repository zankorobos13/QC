
import numpy as np

from functools import lru_cache
from numpy.polynomial.legendre import leggauss

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from operators.fem import shape_functions


# =============================================================
# Caches
# =============================================================

_MATRIX_CACHE = {}
_FACTOR_CACHE = {}


def _array_key(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    return array.shape, array.tobytes()


def _mesh_key(rho_nodes, z_nodes, quadrature_order):
    return (
        _array_key(rho_nodes),
        _array_key(z_nodes),
        int(quadrature_order),
    )


@lru_cache(maxsize=8)
def _quadrature_data(quadrature_order):
    """
    Precompute Gauss points, weights and Q1 shape functions.
    """

    points, weights = leggauss(quadrature_order)

    data = []

    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            N, _, _ = shape_functions(xi, eta)

            data.append((
                N,
                wx * wy,
            ))

    return data


def _mesh_quadrature_data(
    rho_nodes,
    z_nodes,
    quadrature_order,
):
    """
    Precompute element connectivity and quadrature weights.

    Each entry contains:
        N       : Q1 shape functions, shape (4,)
        weights : rho * wx * wy * J_rho * J_z,
                  shape (n_elements,)
    """

    key = _mesh_key(
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    if key in _MESH_CACHE:
        return _MESH_CACHE[key]

    n_rho = len(rho_nodes)
    n_z = len(z_nodes)

    i, j = np.meshgrid(
        np.arange(n_rho - 1),
        np.arange(n_z - 1),
        indexing="ij",
    )

    i = i.ravel()
    j = j.ravel()

    nodes = np.column_stack((
        i * n_z + j,
        (i + 1) * n_z + j,
        i * n_z + j + 1,
        (i + 1) * n_z + j + 1,
    ))

    rho_a = rho_nodes[i]
    rho_b = rho_nodes[i + 1]

    z_a = z_nodes[j]
    z_b = z_nodes[j + 1]

    J_rho = 0.5 * (rho_b - rho_a)
    J_z = 0.5 * (z_b - z_a)

    quadrature_data = []

    for N, quadrature_weight in _quadrature_data(
        quadrature_order
    ):
        points = leggauss(quadrature_order)[0]

        # Recover the reference coordinates from the shape-function
        # evaluation is unnecessary: they are stored below instead.
        quadrature_data.append((N, quadrature_weight))

    # Reconstruct quadrature coordinates explicitly.
    gauss_points, gauss_weights = leggauss(quadrature_order)

    quadrature_data = []

    for xi, wx in zip(gauss_points, gauss_weights):
        for eta, wy in zip(gauss_points, gauss_weights):

            N, _, _ = shape_functions(xi, eta)

            rho = rho_a + 0.5 * (xi + 1.0) * (rho_b - rho_a)

            weights = (
                wx * wy
                * J_rho
                * J_z
                * rho
            )

            quadrature_data.append((N, weights))

    result = {
        "nodes": nodes,
        "quadrature": quadrature_data,
        "n_rho": n_rho,
        "n_z": n_z,
    }

    _MESH_CACHE[key] = result

    return result


_MESH_CACHE = {}


# =============================================================
# Local Poisson matrix
# =============================================================

def assemble_local_poisson_matrix(
    rho_a,
    rho_b,
    z_a,
    z_b,
    k,
    quadrature_order=8,
):
    points, weights = leggauss(quadrature_order)

    J_rho = 0.5 * (rho_b - rho_a)
    J_z = 0.5 * (z_b - z_a)

    K_local = np.zeros((4, 4), dtype=float)

    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):

            rho = J_rho * xi + 0.5 * (rho_a + rho_b)

            N, dN_dxi, dN_deta = shape_functions(xi, eta)

            dN_drho = dN_dxi / J_rho
            dN_dz = dN_deta / J_z

            weight = wx * wy * J_rho * J_z

            K_local += weight * (
                rho * (
                    np.outer(dN_drho, dN_drho)
                    + np.outer(dN_dz, dN_dz)
                )
                + (k**2 / rho) * np.outer(N, N)
            )

    return K_local


# =============================================================
# Global Poisson matrix
# =============================================================

def assemble_poisson_matrix(
    rho_nodes,
    z_nodes,
    k,
    quadrature_order=8,
    return_full=False,
):
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    key = (
        _mesh_key(rho_nodes, z_nodes, quadrature_order),
        int(k),
    )

    if key not in _MATRIX_CACHE:

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
                    k=int(k),
                    quadrature_order=quadrature_order,
                )

                nodes = np.array([
                    i * nz + j,
                    (i + 1) * nz + j,
                    i * nz + j + 1,
                    (i + 1) * nz + j + 1,
                ])

                row, col = np.meshgrid(
                    nodes,
                    nodes,
                    indexing="ij",
                )

                rows.extend(row.ravel())
                cols.extend(col.ravel())
                data.extend(K_local.ravel())

        K_full = coo_matrix(
            (data, (rows, cols)),
            shape=(n_total, n_total),
        ).tocsr()

        free_mask = np.ones(n_total, dtype=bool)

        # Outer radial boundary.
        free_mask[(nr - 1) * nz:nr * nz] = False

        # Lower and upper z boundaries.
        free_mask[np.arange(nr) * nz] = False
        free_mask[np.arange(nr) * nz + nz - 1] = False

        # Axis for nonzero angular momentum.
        if k != 0 and np.isclose(rho_nodes[0], 0.0):
            free_mask[:nz] = False

        free_indices = np.flatnonzero(free_mask)

        K = K_full[free_indices, :][:, free_indices].tocsc()

        _MATRIX_CACHE[key] = (
            K,
            free_indices,
            K_full,
        )

    K, free_indices, K_full = _MATRIX_CACHE[key]

    if return_full:
        return K, free_indices, K_full

    return K, free_indices


# =============================================================
# Cached LU factorization
# =============================================================

def _get_poisson_factorization(
    rho_nodes,
    z_nodes,
    k,
    quadrature_order,
):
    key = (
        _mesh_key(rho_nodes, z_nodes, quadrature_order),
        int(k),
    )

    if key not in _FACTOR_CACHE:

        K, free_indices, K_full = assemble_poisson_matrix(
            rho_nodes,
            z_nodes,
            k,
            quadrature_order,
            return_full=True,
        )

        lu = splu(K)

        _FACTOR_CACHE[key] = (
            lu,
            K,
            free_indices,
            K_full,
        )

    return _FACTOR_CACHE[key]


def _solve_factorized(lu, rhs):
    """
    Solve with a previously computed real LU factorization.
    Complex right-hand sides are handled through real and
    imaginary parts.
    """

    rhs = np.asarray(rhs)

    if np.iscomplexobj(rhs):
        return (
            lu.solve(rhs.real)
            + 1j * lu.solve(rhs.imag)
        )

    return lu.solve(rhs)


# =============================================================
# Vectorized RHS assembly
# =============================================================

def _assemble_rhs_from_local_values(
    local_values,
    mesh_data,
    free_indices,
    quadrature_order,
):
    """
    Assemble:
        b_mu = 2 * integral rho * N_mu * source d(rho) dz

    local_values has shape (n_elements, 4).
    """

    nodes = mesh_data["nodes"]
    n_total = mesh_data["n_rho"] * mesh_data["n_z"]

    dtype = np.result_type(local_values.dtype, np.float64)

    b_full = np.zeros(n_total, dtype=dtype)
    local_rhs = np.zeros((len(nodes), 4), dtype=dtype)

    for N, weights in mesh_data["quadrature"]:

        source = local_values @ N

        local_rhs += (
            2.0
            * weights[:, None]
            * source[:, None]
            * N[None, :]
        )

    np.add.at(
        b_full,
        nodes.ravel(),
        local_rhs.ravel(),
    )

    return b_full[free_indices]


def assemble_poisson_rhs(
    density,
    rho_nodes,
    z_nodes,
    free_indices,
    quadrature_order=8,
):
    density = np.asarray(density)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    free_indices = np.asarray(free_indices, dtype=int)

    expected_shape = (len(rho_nodes), len(z_nodes))

    if density.shape != expected_shape:
        raise ValueError(
            f"density must have shape {expected_shape}, got {density.shape}"
        )

    mesh_data = _mesh_quadrature_data(
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    local_density = density.ravel()[mesh_data["nodes"]]

    return _assemble_rhs_from_local_values(
        local_density,
        mesh_data,
        free_indices,
        quadrature_order,
    )


# =============================================================
# Transition charge
# =============================================================

def _transition_charge(
    orbital_q,
    orbital_s,
    rho_nodes,
    z_nodes,
    quadrature_order,
):
    mesh_data = _mesh_quadrature_data(
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    nodes = mesh_data["nodes"]

    q_local = orbital_q.ravel()[nodes]
    s_local = orbital_s.ravel()[nodes]

    charge = 0.0 + 0.0j

    for N, weights in mesh_data["quadrature"]:

        q_value = q_local @ N
        s_value = s_local @ N

        charge += np.sum(
            weights
            * np.conj(q_value)
            * s_value
        )

    return charge


def _monopole_boundary_values(
    rho_nodes,
    z_nodes,
    charge,
):
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


# =============================================================
# Standard Poisson solver
# =============================================================

def solve_poisson(
    density,
    rho_nodes,
    z_nodes,
    k,
    quadrature_order=8,
):
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

    lu, K, free_indices, K_full = _get_poisson_factorization(
        rho_nodes,
        z_nodes,
        int(k),
        quadrature_order,
    )

    rhs = assemble_poisson_rhs(
        density,
        rho_nodes,
        z_nodes,
        free_indices,
        quadrature_order,
    )

    coefficients = _solve_factorized(lu, rhs)

    potential = np.zeros(
        expected_shape,
        dtype=np.result_type(coefficients.dtype, density.dtype, np.float64),
    )

    potential.ravel()[free_indices] = coefficients

    return potential


# =============================================================
# Vectorized potential integration
# =============================================================

def integrate_poisson_potential(
    density,
    potential,
    rho_nodes,
    z_nodes,
    quadrature_order=8,
):
    density = np.asarray(density)
    potential = np.asarray(potential)

    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    expected_shape = (len(rho_nodes), len(z_nodes))

    if density.shape != expected_shape:
        raise ValueError(
            f"density must have shape {expected_shape}, got {density.shape}"
        )

    if potential.shape != expected_shape:
        raise ValueError(
            f"potential must have shape {expected_shape}, got {potential.shape}"
        )

    mesh_data = _mesh_quadrature_data(
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    nodes = mesh_data["nodes"]

    density_local = density.ravel()[nodes]
    potential_local = potential.ravel()[nodes]

    result = 0.0 + 0.0j if (
        np.iscomplexobj(density) or np.iscomplexobj(potential)
    ) else 0.0

    for N, weights in mesh_data["quadrature"]:

        density_value = density_local @ N
        potential_value = potential_local @ N

        result += np.sum(
            weights
            * density_value
            * potential_value
        )

    return result


# =============================================================
# Vectorized orbital RHS
# =============================================================

def assemble_poisson_rhs_from_orbitals(
    orbital_q,
    orbital_s,
    rho_nodes,
    z_nodes,
    free_indices,
    quadrature_order=8,
):
    orbital_q = np.asarray(orbital_q)
    orbital_s = np.asarray(orbital_s)

    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    expected = (len(rho_nodes), len(z_nodes))

    if orbital_q.shape != expected or orbital_s.shape != expected:
        raise ValueError(
            f"orbital grids must have shape {expected}"
        )

    mesh_data = _mesh_quadrature_data(
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    nodes = mesh_data["nodes"]

    q_local = orbital_q.ravel()[nodes]
    s_local = orbital_s.ravel()[nodes]

    # The product is formed after interpolating each orbital.
    local_rhs = np.zeros(
        (len(nodes), 4),
        dtype=np.result_type(
            orbital_q.dtype,
            orbital_s.dtype,
            np.float64,
        ),
    )

    for N, weights in mesh_data["quadrature"]:

        q_value = q_local @ N
        s_value = s_local @ N

        source = np.conj(q_value) * s_value

        local_rhs += (
            2.0
            * weights[:, None]
            * source[:, None]
            * N[None, :]
        )

    n_total = len(rho_nodes) * len(z_nodes)

    b_full = np.zeros(
        n_total,
        dtype=local_rhs.dtype,
    )

    np.add.at(
        b_full,
        nodes.ravel(),
        local_rhs.ravel(),
    )

    return b_full[free_indices]



def solve_poisson_orbital_pair(
    orbital_q,
    orbital_s,
    rho_nodes,
    z_nodes,
    k,
    quadrature_order=8,
):
    """
    Solve the Poisson equation for one orbital-pair transition density.

    The returned potential can be reused for all integrals having
    the same radial transition density and angular channel.

    Source density:
        rho_qs = conj(psi_q) * psi_s

    Returns
    -------
    potential : ndarray
        Potential on the FEM grid.
    """

    orbital_q = np.asarray(orbital_q)
    orbital_s = np.asarray(orbital_s)

    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")

    if not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer.")

    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive.")

    expected = (len(rho_nodes), len(z_nodes))

    if orbital_q.shape != expected or orbital_s.shape != expected:
        raise ValueError(f"Orbital grids must have shape {expected}.")

    if not np.all(np.isfinite(orbital_q)):
        raise ValueError("orbital_q contains non-finite values.")

    if not np.all(np.isfinite(orbital_s)):
        raise ValueError("orbital_s contains non-finite values.")

    lu, K, free_indices, K_full = _get_poisson_factorization(
        rho_nodes,
        z_nodes,
        int(k),
        quadrature_order,
    )

    rhs = assemble_poisson_rhs_from_orbitals(
        orbital_q,
        orbital_s,
        rho_nodes,
        z_nodes,
        free_indices,
        quadrature_order,
    )

    # Boundary nodes are excluded from the unknown vector.
    boundary_mask = np.ones(expected, dtype=bool)

    boundary_mask[-1, :] = False
    boundary_mask[:, 0] = False
    boundary_mask[:, -1] = False

    if k != 0 and np.isclose(rho_nodes[0], 0.0):
        boundary_mask[0, :] = False

    boundary_indices = np.flatnonzero(
        ~boundary_mask.ravel()
    )

    boundary_values = np.zeros(
        expected,
        dtype=np.result_type(
            orbital_q.dtype,
            orbital_s.dtype,
            np.float64,
        ),
    )

    # Monopole boundary condition for k = 0.
    if k == 0:
        charge = _transition_charge(
            orbital_q,
            orbital_s,
            rho_nodes,
            z_nodes,
            quadrature_order,
        )

        boundary_values = _monopole_boundary_values(
            rho_nodes,
            z_nodes,
            charge,
        )

    boundary_vector = boundary_values.ravel()[boundary_indices]

    rhs = rhs - (
        K_full[free_indices, :][:, boundary_indices]
        @ boundary_vector
    )

    coefficients = _solve_factorized(lu, rhs)

    potential = boundary_values.copy()
    potential.ravel()[free_indices] = coefficients

    return potential


def integrate_orbital_pair_potential(
    orbital_p,
    orbital_r,
    potential,
    rho_nodes,
    z_nodes,
    quadrature_order=8,
):
    """
    Integrate a transition density against a precomputed potential.

    Integral:
        integral conj(psi_p) * psi_r * V * rho d(rho) dz

    The interpolation and quadrature are identical to those used
    in integrate_poisson_orbitals().
    """

    orbital_p = np.asarray(orbital_p)
    orbital_r = np.asarray(orbital_r)
    potential = np.asarray(potential)

    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    expected = (len(rho_nodes), len(z_nodes))

    if orbital_p.shape != expected:
        raise ValueError(f"orbital_p must have shape {expected}.")

    if orbital_r.shape != expected:
        raise ValueError(f"orbital_r must have shape {expected}.")

    if potential.shape != expected:
        raise ValueError(f"potential must have shape {expected}.")

    mesh_data = _mesh_quadrature_data(
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    nodes = mesh_data["nodes"]

    p_local = orbital_p.ravel()[nodes]
    r_local = orbital_r.ravel()[nodes]
    u_local = potential.ravel()[nodes]

    result = 0.0 + 0.0j

    for N, weights in mesh_data["quadrature"]:

        p_value = p_local @ N
        r_value = r_local @ N
        u_value = u_local @ N

        density = np.conj(p_value) * r_value

        result += np.sum(
            weights
            * density
            * u_value
        )

    if not (
        np.iscomplexobj(orbital_p)
        or np.iscomplexobj(orbital_r)
        or np.iscomplexobj(potential)
    ):
        return float(np.real(result))

    return result



# =============================================================
# Main two-electron integral
# =============================================================

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
            raise ValueError(
                f"{name} must have shape {expected}"
            )

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer")

    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")

    # ---------------------------------------------------------
    # 1. Get cached matrix and LU factorization
    # ---------------------------------------------------------

    lu, K, free_indices, K_full = _get_poisson_factorization(
        rho_nodes,
        z_nodes,
        int(k),
        quadrature_order,
    )

    # ---------------------------------------------------------
    # 2. Assemble transition-density RHS
    # ---------------------------------------------------------

    rhs = assemble_poisson_rhs_from_orbitals(
        orbital_q,
        orbital_s,
        rho_nodes,
        z_nodes,
        free_indices,
        quadrature_order,
    )

    # ---------------------------------------------------------
    # 3. Boundary conditions
    # ---------------------------------------------------------

    boundary_mask = np.ones(expected, dtype=bool)

    boundary_mask[-1, :] = False
    boundary_mask[:, 0] = False
    boundary_mask[:, -1] = False

    if k != 0 and np.isclose(rho_nodes[0], 0.0):
        boundary_mask[0, :] = False

    boundary_indices = np.flatnonzero(
        ~boundary_mask.ravel()
    )

    boundary_values = np.zeros(
        expected,
        dtype=np.result_type(
            orbital_q.dtype,
            orbital_s.dtype,
            np.float64,
        ),
    )

    if k == 0:

        charge = _transition_charge(
            orbital_q,
            orbital_s,
            rho_nodes,
            z_nodes,
            quadrature_order,
        )

        boundary_values = _monopole_boundary_values(
            rho_nodes,
            z_nodes,
            charge,
        )

    boundary_vector = boundary_values.ravel()[boundary_indices]

    rhs = rhs - (
        K_full[free_indices, :][:, boundary_indices]
        @ boundary_vector
    )

    # ---------------------------------------------------------
    # 4. Solve with cached LU
    # ---------------------------------------------------------

    coefficients = _solve_factorized(lu, rhs)

    potential = boundary_values.copy()
    potential.ravel()[free_indices] = coefficients

    # ---------------------------------------------------------
    # 5. Integrate transition density times potential
    # ---------------------------------------------------------

    mesh_data = _mesh_quadrature_data(
        rho_nodes,
        z_nodes,
        quadrature_order,
    )

    nodes = mesh_data["nodes"]

    p_local = orbital_p.ravel()[nodes]
    r_local = orbital_r.ravel()[nodes]
    u_local = potential.ravel()[nodes]

    result = 0.0 + 0.0j

    for N, weights in mesh_data["quadrature"]:

        p_value = p_local @ N
        r_value = r_local @ N
        u_value = u_local @ N

        density_pr = np.conj(p_value) * r_value

        result += np.sum(
            weights
            * density_pr
            * u_value
        )

    if (
        not np.iscomplexobj(orbital_p)
        and not np.iscomplexobj(orbital_q)
        and not np.iscomplexobj(orbital_r)
        and not np.iscomplexobj(orbital_s)
    ):
        return float(np.real(result))

    return result
