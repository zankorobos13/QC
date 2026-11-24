
import numpy as np


def _validate_hermitian_matrix(matrix, shape, name, tolerance=1e-9):
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


def calculate_electronic_energy(
    h_alpha,
    h_beta,
    F_alpha,
    F_beta,
    D_alpha,
    D_beta,
    hermiticity_tolerance=1e-9,
):
    """
    Calculate the UHF electronic energy.

    E = 1/2 Tr[D_alpha (h_alpha + F_alpha)]
      + 1/2 Tr[D_beta  (h_beta  + F_beta)]

    The density convention is:

        D[p, q] = sum_i C[p, i] * conj(C[q, i])

    Returns
    -------
    float
        Real electronic energy in Hartree.
    """

    h_alpha = np.asarray(h_alpha, dtype=np.complex128)

    if h_alpha.ndim != 2 or h_alpha.shape[0] != h_alpha.shape[1]:
        raise ValueError("h_alpha must be a square matrix.")

    n = h_alpha.shape[0]
    shape = (n, n)

    h_alpha = _validate_hermitian_matrix(
        h_alpha, shape, "h_alpha", hermiticity_tolerance
    )
    h_beta = _validate_hermitian_matrix(
        h_beta, shape, "h_beta", hermiticity_tolerance
    )
    F_alpha = _validate_hermitian_matrix(
        F_alpha, shape, "F_alpha", hermiticity_tolerance
    )
    F_beta = _validate_hermitian_matrix(
        F_beta, shape, "F_beta", hermiticity_tolerance
    )
    D_alpha = _validate_hermitian_matrix(
        D_alpha, shape, "D_alpha", hermiticity_tolerance
    )
    D_beta = _validate_hermitian_matrix(
        D_beta, shape, "D_beta", hermiticity_tolerance
    )

    E_alpha = 0.5 * np.einsum(
        "pq,qp->",
        D_alpha,
        h_alpha + F_alpha,
        optimize=True,
    )

    E_beta = 0.5 * np.einsum(
        "pq,qp->",
        D_beta,
        h_beta + F_beta,
        optimize=True,
    )

    energy = E_alpha + E_beta

    if abs(energy.imag) > hermiticity_tolerance:
        raise ValueError(
            "Electronic energy has a non-negligible imaginary part: "
            f"{energy.imag}."
        )

    return float(energy.real)
