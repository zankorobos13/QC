import numpy as np

from scipy.sparse import coo_matrix

from numpy.polynomial.legendre import leggauss

def shape_functions(xi, eta):
    """
    Bilinear Q1 basis on the reference square [-1, 1] x [-1, 1].

    Local node order:
        0: (-1, -1)
        1: (+1, -1)
        2: (-1, +1)
        3: (+1, +1)
    """

    N = 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 - xi) * (1.0 + eta),
        (1.0 + xi) * (1.0 + eta),
    ])

    dN_dxi = 0.25 * np.array([
        -(1.0 - eta),
         (1.0 - eta),
        -(1.0 + eta),
         (1.0 + eta),
    ])

    dN_deta = 0.25 * np.array([
        -(1.0 - xi),
        -(1.0 + xi),
         (1.0 - xi),
         (1.0 + xi),
    ])

    return N, dN_dxi, dN_deta


def assemble_fem_matrices(rho_nodes, z_nodes, Z, B, m, spin=0.0, quadrature_order=8):
    """
    Assemble the generalized eigenvalue problem

        (T + V) c = E S c

    using bilinear Q1 finite elements.

    Parameters
    ----------
    rho_nodes : array
        Physical radial mesh nodes, starting at zero.
    z_nodes : array
        Physical z mesh nodes.
    Z : float
        Nuclear charge.
    B : float
        Magnetic field in atomic units.
    m : int
        Magnetic quantum number in the convention
        Psi ~ exp(-i*m*phi).
    spin : float
        Spin projection: +0.5, -0.5, or 0.0.

    Returns
    -------
    T, S, V : sparse CSC matrices
        Matrices after applying boundary conditions.
    free_indices : ndarray
        Indices of retained global degrees of freedom.
    """

    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    if np.any(np.diff(rho_nodes) <= 0):
        raise ValueError("Radial nodes must be strictly increasing.")

    if np.any(np.diff(z_nodes) <= 0):
        raise ValueError("z nodes must be strictly increasing.")

    if rho_nodes[0] < 0:
        raise ValueError("Radial coordinates cannot be negative.")

    if Z <= 0:
        raise ValueError("Nuclear charge must be positive.")

    if not isinstance(m, (int, np.integer)):
        raise TypeError("m must be an integer.")

    nr = len(rho_nodes)
    nz = len(z_nodes)
    n_total = nr * nz

    # Gauss-Legendre quadrature on [-1, 1].
    gauss_points, gauss_weights = leggauss(quadrature_order)

    rows = []
    cols = []

    t_data = []
    s_data = []
    v_data = []

    for i in range(nr - 1):
        rho_left = rho_nodes[i]
        rho_right = rho_nodes[i + 1]
        dr = rho_right - rho_left

        for j in range(nz - 1):
            z_bottom = z_nodes[j]
            z_top = z_nodes[j + 1]
            dz = z_top - z_bottom

            det_jacobian = dr * dz / 4.0

            # Global indices of the four local nodes.
            nodes = np.array([
                i * nz + j,
                (i + 1) * nz + j,
                i * nz + (j + 1),
                (i + 1) * nz + (j + 1),
            ])

            local_T = np.zeros((4, 4))
            local_S = np.zeros((4, 4))
            local_V = np.zeros((4, 4))

            for xi, wx in zip(gauss_points, gauss_weights):
                for eta, wy in zip(gauss_points, gauss_weights):

                    N, dN_dxi, dN_deta = shape_functions(xi, eta)

                    rho = (
                        rho_left
                        + 0.5 * (xi + 1.0) * dr
                    )

                    z = (
                        z_bottom
                        + 0.5 * (eta + 1.0) * dz
                    )

                    dN_drho = dN_dxi * (2.0 / dr)
                    dN_dz = dN_deta * (2.0 / dz)

                    weight = wx * wy * det_jacobian

                    # Cylindrical volume element: rho d(rho) dz.
                    measure = rho * weight

                    local_T += 0.5 * weight * (
                        rho * (
                            np.outer(dN_drho, dN_drho)
                            + np.outer(dN_dz, dN_dz)
                        )
                        + (m * m / rho) * np.outer(N, N)
                    )

                    # Overlap matrix.
                    local_S += measure * np.outer(N, N)

                    # One-electron potential.
                    r = np.sqrt(rho * rho + z * z)

                    V_coulomb = -Z / r
                    V_magnetic = (
                        -0.5 * B * m
                        + 0.125 * B * B * rho * rho
                        + B * spin
                    )

                    V_local = V_coulomb + V_magnetic

                    local_V += (
                        measure
                        * V_local
                        * np.outer(N, N)
                    )

            for a in range(4):
                for b in range(4):
                    rows.append(nodes[a])
                    cols.append(nodes[b])

                    t_data.append(local_T[a, b])
                    s_data.append(local_S[a, b])
                    v_data.append(local_V[a, b])

    shape = (n_total, n_total)

    T_full = coo_matrix(
        (t_data, (rows, cols)), shape=shape
    ).tocsc()

    S_full = coo_matrix(
        (s_data, (rows, cols)), shape=shape
    ).tocsc()

    V_full = coo_matrix(
        (v_data, (rows, cols)), shape=shape
    ).tocsc()

    # Boundary conditions:
    # psi = 0 at outer radial boundary and both z boundaries.
    #
    # For m != 0, regularity requires psi = 0 at rho = 0.
    # For m == 0, the axis is retained and has a natural
    # regularity boundary condition.

    free_mask = np.ones((nr, nz), dtype=bool)

    free_mask[-1, :] = False
    free_mask[:, 0] = False
    free_mask[:, -1] = False

    if m != 0:
        free_mask[0, :] = False

    free_indices = np.flatnonzero(free_mask.ravel())

    T = T_full[free_indices, :][:, free_indices].tocsc()
    S = S_full[free_indices, :][:, free_indices].tocsc()
    V = V_full[free_indices, :][:, free_indices].tocsc()

    return T, S, V, free_indices
