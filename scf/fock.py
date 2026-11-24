
import numpy as np


def _validate_matrix(matrix, shape, name, tolerance=1e-10):
    matrix = np.asarray(matrix, dtype=np.complex128)

    if matrix.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")

    if not np.allclose(
        matrix,
        matrix.conj().T,
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must be Hermitian.")

    return matrix


def build_coulomb_exchange(
    g,
    D_alpha,
    D_beta,
    hermiticity_tolerance=1e-9,
):
    """
    Build Coulomb and exchange matrices for UHF.

    Integral convention
    -------------------
    g[p, q, r, s] = <p q | 1/r12 | r s>

    Density convention
    ------------------
    D[p, q] = sum_i C[p, i] * conj(C[q, i])

    Returns
    -------
    J : ndarray
        Coulomb matrix.

    K_alpha : ndarray
        Alpha-spin exchange matrix.

    K_beta : ndarray
        Beta-spin exchange matrix.
    """

    g = np.asarray(g, dtype=np.complex128)

    if g.ndim != 4:
        raise ValueError("g must be a four-dimensional tensor.")

    n = g.shape[0]

    if g.shape != (n, n, n, n):
        raise ValueError("g must have shape (n, n, n, n).")

    if not np.all(np.isfinite(g)):
        raise ValueError("g contains non-finite values.")

    expected_shape = (n, n)

    D_alpha = _validate_matrix(
        D_alpha, expected_shape, "D_alpha"
    )
    D_beta = _validate_matrix(
        D_beta, expected_shape, "D_beta"
    )

    if not np.isfinite(hermiticity_tolerance) or (
        hermiticity_tolerance <= 0
    ):
        raise ValueError(
            "hermiticity_tolerance must be positive and finite."
        )

    D_total = D_alpha + D_beta

    # J[a, b] = sum_rs D_total[r, s] * g[a, r, b, s]
    J = np.einsum(
        "rs,arbs->ab",
        D_total,
        g,
        optimize=True,
    )

    # K[a, b] = sum_rs D[r, s] * g[a, s, r, b]
    K_alpha = np.einsum(
        "rs,asrb->ab",
        D_alpha,
        g,
        optimize=True,
    )

    K_beta = np.einsum(
        "rs,asrb->ab",
        D_beta,
        g,
        optimize=True,
    )

    # Remove numerical anti-Hermitian noise.
    J = 0.5 * (J + J.conj().T)
    K_alpha = 0.5 * (K_alpha + K_alpha.conj().T)
    K_beta = 0.5 * (K_beta + K_beta.conj().T)

    return J, K_alpha, K_beta


def build_fock_matrices(
    h_alpha,
    h_beta,
    g,
    D_alpha,
    D_beta,
    hermiticity_tolerance=1e-9,
):
    """
    Build alpha- and beta-spin Fock matrices.

    F_alpha = h_alpha + J - K_alpha
    F_beta  = h_beta  + J - K_beta

    Returns
    -------
    F_alpha, F_beta, J, K_alpha, K_beta
    """

    h_alpha = np.asarray(h_alpha, dtype=np.complex128)
    h_beta = np.asarray(h_beta, dtype=np.complex128)

    if h_alpha.ndim != 2 or h_alpha.shape[0] != h_alpha.shape[1]:
        raise ValueError("h_alpha must be a square matrix.")

    n = h_alpha.shape[0]
    expected_shape = (n, n)

    h_alpha = _validate_matrix(
        h_alpha,
        expected_shape,
        "h_alpha",
        hermiticity_tolerance,
    )

    h_beta = _validate_matrix(
        h_beta,
        expected_shape,
        "h_beta",
        hermiticity_tolerance,
    )

    J, K_alpha, K_beta = build_coulomb_exchange(
        g,
        D_alpha,
        D_beta,
        hermiticity_tolerance,
    )

    F_alpha = h_alpha + J - K_alpha
    F_beta = h_beta + J - K_beta

    # Enforce Hermiticity against numerical noise.
    F_alpha = 0.5 * (F_alpha + F_alpha.conj().T)
    F_beta = 0.5 * (F_beta + F_beta.conj().T)

    return F_alpha, F_beta, J, K_alpha, K_beta
