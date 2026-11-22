import numpy as np
import pytest

from integrals.coulomb import (
    build_full_coefficients,
    transition_density,
    angular_coulomb_kernel,
    integrate_transition_densities,
    integrate,
)


# ============================================================
# Test configuration
# ============================================================

RHO_NODES = np.array([0.5, 1.0])
Z_NODES = np.array([-0.5, 0.5])

F = np.ones((2, 2), dtype=complex)

QUADRATURE_ORDER = 2
ANGULAR_ORDER = 16


# ============================================================
# 1. Full coefficient reconstruction
# ============================================================

def test_build_full_coefficients():

    coefficients = np.array([10, 20, 30])
    free_indices = np.array([0, 2, 4])

    result = build_full_coefficients(
        coefficients=coefficients,
        free_indices=free_indices,
        n_total=5,
    )

    expected = np.array([10, 0, 20, 0, 30])

    np.testing.assert_array_equal(result, expected)


# ============================================================
# 2. Transition density
# ============================================================

def test_transition_density():

    psi_p = np.array([
        [1.0 + 1.0j, 2.0 + 0.0j],
    ])

    psi_r = np.array([
        [2.0 + 0.0j, 1.0 + 1.0j],
    ])

    result = transition_density(psi_p, psi_r)

    expected = np.array([
        [2.0 - 2.0j, 2.0 + 2.0j],
    ])

    np.testing.assert_allclose(result, expected)


def test_transition_density_requires_equal_shapes():

    with pytest.raises(ValueError):
        transition_density(
            np.ones((2, 2)),
            np.ones((3, 3)),
        )


# ============================================================
# 3. Angular Coulomb kernel
# ============================================================

@pytest.mark.parametrize("k", [0, 1, 2, -1, -2])
def test_angular_kernel_is_finite(k):

    result = angular_coulomb_kernel(
        rho1=0.5,
        z1=-0.2,
        rho2=0.8,
        z2=0.4,
        k=k,
        quadrature_order=ANGULAR_ORDER,
    )

    assert np.isfinite(result)


def test_angular_kernel_is_positive_for_k_zero():

    result = angular_coulomb_kernel(
        0.5, -0.2,
        0.8, 0.4,
        k=0,
        quadrature_order=ANGULAR_ORDER,
    )

    assert result > 0.0


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_angular_kernel_is_invariant_under_sign_change(k):

    positive = angular_coulomb_kernel(
        0.5, -0.2,
        0.8, 0.4,
        k=k,
        quadrature_order=ANGULAR_ORDER,
    )

    negative = angular_coulomb_kernel(
        0.5, -0.2,
        0.8, 0.4,
        k=-k,
        quadrature_order=ANGULAR_ORDER,
    )

    np.testing.assert_allclose(
        positive,
        negative,
        atol=1e-12,
    )


def test_angular_kernel_requires_integer_k():

    with pytest.raises(TypeError):
        angular_coulomb_kernel(
            0.5, 0.0,
            0.8, 0.0,
            k=0.5,
        )


# ============================================================
# 4. Four-dimensional integration
# ============================================================

def test_integrate_transition_densities_is_finite():

    result = integrate_transition_densities(
        f_pr=F,
        f_qs=F,
        rho_nodes=RHO_NODES,
        z_nodes=Z_NODES,
        k=0,
        quadrature_order=QUADRATURE_ORDER,
        angular_quadrature_order=ANGULAR_ORDER,
    )

    assert np.isfinite(result)
    assert np.real(result) > 0.0


def test_integral_is_symmetric_under_electron_exchange():

    f1 = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ], dtype=complex)

    f2 = np.array([
        [2.0, 1.0],
        [0.5, 3.0],
    ], dtype=complex)

    first = integrate_transition_densities(
        f1, f2,
        RHO_NODES, Z_NODES,
        k=1,
        quadrature_order=QUADRATURE_ORDER,
        angular_quadrature_order=ANGULAR_ORDER,
    )

    second = integrate_transition_densities(
        f2, f1,
        RHO_NODES, Z_NODES,
        k=-1,
        quadrature_order=QUADRATURE_ORDER,
        angular_quadrature_order=ANGULAR_ORDER,
    )

    np.testing.assert_allclose(
        first,
        second,
        atol=1e-10,
        rtol=1e-10,
    )


def test_integral_hermiticity():

    f1 = np.array([
        [1.0 + 1.0j, 2.0 - 0.5j],
        [0.5 + 0.2j, 1.0 - 1.0j],
    ])

    f2 = np.array([
        [2.0 - 0.3j, 1.0 + 0.7j],
        [1.5 + 0.2j, 0.5 - 0.1j],
    ])

    first = integrate_transition_densities(
        f1, f2,
        RHO_NODES, Z_NODES,
        k=1,
        quadrature_order=QUADRATURE_ORDER,
        angular_quadrature_order=ANGULAR_ORDER,
    )

    second = integrate_transition_densities(
        f1.conj(), f2.conj(),
        RHO_NODES, Z_NODES,
        k=-1,
        quadrature_order=QUADRATURE_ORDER,
        angular_quadrature_order=ANGULAR_ORDER,
    )

    np.testing.assert_allclose(
        second,
        np.conj(first),
        atol=1e-10,
        rtol=1e-10,
    )


# ============================================================
# 5. Selection rule
# ============================================================

def test_integrate_selection_rule():

    result = integrate(
        orbital_p=F,
        orbital_q=F,
        orbital_r=F,
        orbital_s=F,
        rho_nodes=RHO_NODES,
        z_nodes=Z_NODES,
        m_p=1,
        m_q=0,
        m_r=0,
        m_s=0,
    )

    assert result == 0.0


def test_integrate_allowed_selection_rule():

    result = integrate(
        orbital_p=F,
        orbital_q=F,
        orbital_r=F,
        orbital_s=F,
        rho_nodes=RHO_NODES,
        z_nodes=Z_NODES,
        m_p=1,
        m_q=0,
        m_r=0,
        m_s=1,
        quadrature_order=QUADRATURE_ORDER,
        angular_quadrature_order=ANGULAR_ORDER,
    )

    assert np.isfinite(result)