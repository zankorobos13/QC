
import numpy as np


def build_density_matrix(
    coefficients,
    n_occupied,
    orthonormality_tolerance=1e-9,
):
    """
    Build a one-particle density matrix from occupied orbitals.

    Parameters
    ----------
    coefficients : ndarray, shape (n_orbitals, n_mo)
        Molecular orbital coefficient matrix. Occupied orbitals
        are stored in the first n_occupied columns.

    n_occupied : int
        Number of occupied orbitals.

    orthonormality_tolerance : float
        Tolerance for checking orthonormality of occupied orbitals.

    Returns
    -------
    density : ndarray, shape (n_orbitals, n_orbitals)
        One-particle density matrix:

            D[p, q] = sum_i C[p, i] * conj(C[q, i])
    """

    coefficients = np.asarray(coefficients, dtype=np.complex128)

    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a two-dimensional array.")

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients contain non-finite values.")

    if isinstance(n_occupied, bool) or not isinstance(
        n_occupied, (int, np.integer)
    ):
        raise TypeError("n_occupied must be an integer.")

    n_orbitals, n_mo = coefficients.shape

    if not 0 <= n_occupied <= n_mo:
        raise ValueError(
            "n_occupied must be between 0 and the number of "
            "available molecular orbitals."
        )

    if not np.isfinite(orthonormality_tolerance) or (
        orthonormality_tolerance <= 0
    ):
        raise ValueError(
            "orthonormality_tolerance must be positive and finite."
        )

    occupied_coefficients = coefficients[:, :n_occupied]

    # Check orthonormality of occupied molecular orbitals.
    overlap = occupied_coefficients.conj().T @ occupied_coefficients

    expected_overlap = np.eye(n_occupied, dtype=np.complex128)

    if not np.allclose(
        overlap,
        expected_overlap,
        atol=orthonormality_tolerance,
        rtol=0.0,
    ):
        raise ValueError("Occupied orbitals are not orthonormal.")

    # D[p, q] = sum_i C[p, i] * conj(C[q, i]).
    density = occupied_coefficients @ occupied_coefficients.conj().T

    # Remove possible numerical anti-Hermitian noise.
    density = 0.5 * (density + density.conj().T)

    return density


def build_spin_densities(
    coefficients_alpha,
    coefficients_beta,
    n_alpha,
    n_beta,
    orthonormality_tolerance=1e-9,
):
    """
    Build alpha- and beta-spin density matrices.

    Returns
    -------
    D_alpha : ndarray
        Alpha-spin density matrix.

    D_beta : ndarray
        Beta-spin density matrix.
    """

    coefficients_alpha = np.asarray(
        coefficients_alpha,
        dtype=np.complex128,
    )
    coefficients_beta = np.asarray(
        coefficients_beta,
        dtype=np.complex128,
    )

    if coefficients_alpha.ndim != 2 or coefficients_beta.ndim != 2:
        raise ValueError("Coefficient matrices must be two-dimensional.")

    if coefficients_alpha.shape[0] != coefficients_beta.shape[0]:
        raise ValueError(
            "Alpha and beta coefficient matrices must have "
            "the same number of spatial basis functions."
        )

    D_alpha = build_density_matrix(
        coefficients_alpha,
        n_alpha,
        orthonormality_tolerance,
    )

    D_beta = build_density_matrix(
        coefficients_beta,
        n_beta,
        orthonormality_tolerance,
    )

    return D_alpha, D_beta
