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

def _all_free_indices(rho_nodes, z_nodes):
    return np.arange(len(rho_nodes) * len(z_nodes))


def test_poisson_rhs_constant_density_matches_integral():
    from integrals.poisson import assemble_poisson_rhs

    rho_nodes = np.array([0.0, 0.4, 1.0])
    z_nodes = np.array([-1.0, 0.0, 1.0])
    density = np.ones((len(rho_nodes), len(z_nodes)))
    free = _all_free_indices(rho_nodes, z_nodes)

    rhs = assemble_poisson_rhs(
        density, rho_nodes, z_nodes, free, quadrature_order=4
    )

    # Partition of unity: sum_mu b_mu = 2 * integral rho d(rho) dz.
    expected = (
        2.0
        * 0.5 * (rho_nodes[-1] ** 2 - rho_nodes[0] ** 2)
        * (z_nodes[-1] - z_nodes[0])
    )
    np.testing.assert_allclose(np.sum(rhs), expected, atol=1e-12)


def test_poisson_rhs_supports_complex_density():
    from integrals.poisson import assemble_poisson_rhs

    rho_nodes = np.array([0.0, 0.5, 1.0])
    z_nodes = np.array([-1.0, 0.0, 1.0])
    density = np.ones((3, 3), dtype=complex) * (1.0 + 2.0j)
    free = _all_free_indices(rho_nodes, z_nodes)

    rhs = assemble_poisson_rhs(
        density, rho_nodes, z_nodes, free, quadrature_order=3
    )

    assert np.iscomplexobj(rhs)
    assert np.all(np.abs(rhs.imag - 2.0 * rhs.real) < 1e-12)


def test_poisson_rhs_restricts_to_free_indices():
    from integrals.poisson import assemble_poisson_rhs

    rho_nodes = np.array([0.0, 0.5, 1.0])
    z_nodes = np.array([-1.0, 0.0, 1.0])
    density = np.ones((3, 3))
    free = np.array([1, 2, 4, 5, 7])

    rhs_restricted = assemble_poisson_rhs(
        density, rho_nodes, z_nodes, free, quadrature_order=3
    )
    rhs_full = assemble_poisson_rhs(
        density, rho_nodes, z_nodes,
        _all_free_indices(rho_nodes, z_nodes),
        quadrature_order=3,
    )

    np.testing.assert_allclose(rhs_restricted, rhs_full[free])


def test_poisson_rhs_rejects_wrong_density_shape():
    from integrals.poisson import assemble_poisson_rhs

    with pytest.raises(ValueError, match="density must have shape"):
        assemble_poisson_rhs(
            np.ones((2, 2)),
            np.array([0.0, 0.5, 1.0]),
            np.array([-1.0, 0.0, 1.0]),
            np.arange(9),
        )


def test_solve_poisson_zero_density_returns_zero():
    from integrals.poisson import solve_poisson

    rho_nodes = np.linspace(0.0, 2.0, 6)
    z_nodes = np.linspace(-2.0, 2.0, 7)
    density = np.zeros((len(rho_nodes), len(z_nodes)))

    potential = solve_poisson(
        density, rho_nodes, z_nodes, k=0, quadrature_order=3
    )

    assert potential.shape == density.shape
    np.testing.assert_allclose(potential, 0.0, atol=1e-13)


@pytest.mark.parametrize("k", [0, 1, -1, 2, -2])
def test_solve_poisson_boundary_conditions(k):
    from integrals.poisson import solve_poisson

    rho_nodes = np.linspace(0.0, 2.0, 6)
    z_nodes = np.linspace(-2.0, 2.0, 7)
    density = np.ones((len(rho_nodes), len(z_nodes)))

    potential = solve_poisson(
        density, rho_nodes, z_nodes, k=k, quadrature_order=3
    )

    np.testing.assert_allclose(potential[-1, :], 0.0, atol=1e-13)
    np.testing.assert_allclose(potential[:, 0], 0.0, atol=1e-13)
    np.testing.assert_allclose(potential[:, -1], 0.0, atol=1e-13)
    if k != 0:
        np.testing.assert_allclose(potential[0, :], 0.0, atol=1e-13)


def test_solve_poisson_supports_complex_density():
    from integrals.poisson import solve_poisson

    rho_nodes = np.linspace(0.0, 2.0, 6)
    z_nodes = np.linspace(-2.0, 2.0, 7)
    density = np.ones((6, 7), dtype=complex) * (1.0 + 2.0j)

    potential = solve_poisson(
        density, rho_nodes, z_nodes, k=0, quadrature_order=3
    )

    assert np.iscomplexobj(potential)
    np.testing.assert_allclose(
        potential.imag, 2.0 * potential.real, atol=1e-11
    )


def test_solve_poisson_rejects_noninteger_k():
    from integrals.poisson import solve_poisson

    with pytest.raises(TypeError, match="k must be an integer"):
        solve_poisson(
            np.ones((3, 3)),
            np.array([0.0, 0.5, 1.0]),
            np.array([-1.0, 0.0, 1.0]),
            k=0.5,
        )
