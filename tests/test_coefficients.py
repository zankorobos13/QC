import numpy as np

from grid.mesh import make_atomic_mesh
from operators.coefficients import (
    build_grid_coefficients,
    regular_magnetic_potential,
)


def make_test_coefficients(m=0):
    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=20,
        n_z=24,
        A_rho=1.0,
        A_z=1.0,
        b=0.5,
        t_rho_max=0.95,
        t_z_max=0.95,
    )

    return build_grid_coefficients(rho_mesh, z_mesh, m)


def test_array_shapes():
    c = make_test_coefficients()

    assert c.rho.shape == (21, 1)
    assert c.z.shape == (1, 25)

    assert c.measure.shape == (21, 25)
    assert c.kinetic_rho.shape == (21, 25)
    assert c.kinetic_z.shape == (21, 25)


def test_jacobians_are_positive():
    c = make_test_coefficients()

    assert np.all(c.jacobian_rho > 0)
    assert np.all(c.jacobian_z > 0)


def test_measure_is_nonnegative():
    c = make_test_coefficients()

    assert np.all(c.measure >= 0)


def test_kinetic_coefficients_are_nonnegative():
    c = make_test_coefficients()

    assert np.all(c.kinetic_rho >= 0)
    assert np.all(c.kinetic_z >= 0)


def test_centrifugal_term_for_m_zero():
    c = make_test_coefficients(m=0)

    assert np.all(c.centrifugal == 0)


def test_centrifugal_term_for_nonzero_m():
    c = make_test_coefficients(m=1)

    assert np.all(np.isinf(c.centrifugal[0, :]))
    assert np.all(c.centrifugal[1:, :] > 0)


def test_magnetic_potential():
    rho = np.array([[0.0], [1.0], [2.0]])

    m = 1
    B = 2.0

    V = regular_magnetic_potential(rho, m, B)

    expected = -1.0 + 0.5 * rho**2

    assert np.allclose(V, expected)


def test_quadrature_weights():
    c = make_test_coefficients()

    assert np.all(c.quadrature_weights >= 0)
    assert c.quadrature_weights.shape == (21, 25)
