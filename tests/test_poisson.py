import numpy as np
import pytest

from integrals.poisson import assemble_local_poisson_matrix


@pytest.mark.parametrize("k", [0, 1, -1, 2, -2])
def test_poisson_matrix_shape(k):
    K = assemble_local_poisson_matrix(
        rho_a=0.5,
        rho_b=1.5,
        z_a=-1.0,
        z_b=1.0,
        k=k,
    )

    assert K.shape == (4, 4)


@pytest.mark.parametrize("k", [0, 1, -1, 2, -2])
def test_poisson_matrix_symmetry(k):
    K = assemble_local_poisson_matrix(
        rho_a=0.5,
        rho_b=1.5,
        z_a=-1.0,
        z_b=1.0,
        k=k,
    )

    np.testing.assert_allclose(K, K.T, atol=1e-12)


@pytest.mark.parametrize("k", [0, 1, -1, 2, -2])
def test_poisson_matrix_positive_semidefinite(k):
    K = assemble_local_poisson_matrix(
        rho_a=0.5,
        rho_b=1.5,
        z_a=-1.0,
        z_b=1.0,
        k=k,
    )

    eigenvalues = np.linalg.eigvalsh(K)

    assert np.all(eigenvalues >= -1e-12)


def test_poisson_matrix_k_sign_invariance():
    K_positive = assemble_local_poisson_matrix(
        0.5, 1.5, -1.0, 1.0, k=2
    )

    K_negative = assemble_local_poisson_matrix(
        0.5, 1.5, -1.0, 1.0, k=-2
    )

    np.testing.assert_allclose(
        K_positive,
        K_negative,
        atol=1e-12,
    )


def test_poisson_matrix_k_zero_constant_mode():
    K = assemble_local_poisson_matrix(
        0.5, 1.5, -1.0, 1.0, k=0
    )

    constant = np.ones(4)

    np.testing.assert_allclose(
        K @ constant,
        np.zeros(4),
        atol=1e-12,
    )


def test_poisson_matrix_increases_with_k_squared():
    K0 = assemble_local_poisson_matrix(
        0.5, 1.5, -1.0, 1.0, k=0
    )

    K1 = assemble_local_poisson_matrix(
        0.5, 1.5, -1.0, 1.0, k=1
    )

    assert np.all(np.linalg.eigvalsh(K1 - K0) >= -1e-12)