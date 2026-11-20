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
    quadrature_order=8
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

    return K, free_indices