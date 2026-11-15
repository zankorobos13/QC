import numpy as np

from grid.mesh import (
    atomic_poisson_map,
    atomic_poisson_jacobian,
    make_mesh,
    make_atomic_mesh
)


def test_origin():
    assert atomic_poisson_map(0.0, 1.0, 0.5) == 0.0


def test_symmetry():
    x = np.linspace(-0.9, 0.9, 101)

    positive = atomic_poisson_map(x, 1.0, 0.5)
    negative = atomic_poisson_map(-x, 1.0, 0.5)

    assert np.allclose(positive, -negative)


def test_monotonicity():
    mesh = make_mesh(100, 1.0, 0.5, -0.95, 0.95)

    assert np.all(np.diff(mesh.physical) > 0)


def test_jacobian():
    t = np.linspace(-0.9, 0.9, 100)

    numerical = (
        atomic_poisson_map(t + 1e-7, 1.0, 0.5)
        - atomic_poisson_map(t - 1e-7, 1.0, 0.5)
    ) / (2e-7)

    analytical = atomic_poisson_jacobian(t, 1.0, 0.5)

    assert np.allclose(numerical, analytical, rtol=1e-5)


def test_positive_jacobian():
    t = np.linspace(-0.999, 0.999, 1000)

    J = atomic_poisson_jacobian(t, 1.0, 0.5)

    assert np.all(J > 0)


def test_cylindrical_mesh():
    rho, z = make_atomic_mesh(
        80, 80,
        1.0, 1.0,
        0.5,
        0.95, 0.95
    )

    assert rho.physical[0] == 0.0
    assert np.isclose(z.physical[0], -z.physical[-1])
    assert np.all(np.diff(rho.physical) > 0)
    assert np.all(np.diff(z.physical) > 0)