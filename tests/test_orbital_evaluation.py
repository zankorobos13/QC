import numpy as np
import pytest

from scipy.sparse import csc_matrix, diags

import orbitals.solver as solver
from orbitals.solver import OrbitalSet, solve_orbitals


@pytest.fixture
def mock_fem(monkeypatch):
    """
    Replace FEM assembly with small diagonal matrices.

    Each m sector receives a different Hamiltonian.
    """

    n_fem = 6

    S = diags(np.ones(n_fem), format="csc")
    T = diags(np.linspace(1.0, 2.0, n_fem), format="csc")

    def assemble(
        rho_nodes,
        z_nodes,
        Z,
        B,
        m,
        spin,
        quadrature_order,
    ):
        # Sector-dependent potential.
        V = diags(
            np.linspace(-2.0, -1.0, n_fem) - 0.1 * abs(m),
            format="csc",
        )

        return T, S, V, np.arange(n_fem)

    monkeypatch.setattr(
        solver,
        "assemble_fem_matrices",
        assemble,
    )


@pytest.fixture
def solver_args():
    return {
        "rho_nodes": np.linspace(0.0, 5.0, 5),
        "z_nodes": np.linspace(-5.0, 5.0, 5),
        "Z": 2.0,
        "B": 0.1,
    }


def test_returns_list_of_orbital_sets(mock_fem, solver_args):
    """One OrbitalSet must be returned for each m sector."""

    result = solve_orbitals(
        **solver_args,
        m=np.array([-1, 0, 1]),
        n_orbitals=np.array([2, 3, 2]),
    )

    assert isinstance(result, list)
    assert len(result) == 3

    assert all(isinstance(item, OrbitalSet) for item in result)


def test_magnetic_quantum_numbers(mock_fem, solver_args):
    """Each result must correspond to its requested m."""

    m = np.array([-2, -1, 0, 1, 2])
    n_orbitals = np.array([2, 2, 2, 2, 2])

    result = solve_orbitals(
        **solver_args,
        m=m,
        n_orbitals=n_orbitals,
    )

    assert [item.m for item in result] == list(m)


def test_number_of_orbitals_per_sector(mock_fem, solver_args):
    """Each sector must contain the requested number of orbitals."""

    m = np.array([-1, 0, 1])
    n_orbitals = np.array([1, 3, 2])

    result = solve_orbitals(
        **solver_args,
        m=m,
        n_orbitals=n_orbitals,
    )

    for orbital_set, expected in zip(result, n_orbitals):
        assert orbital_set.n_orbitals == expected


def test_coefficient_shapes(mock_fem, solver_args):
    """Check coefficient matrix dimensions."""

    m = np.array([-1, 0, 1])
    n_orbitals = np.array([2, 3, 1])

    result = solve_orbitals(
        **solver_args,
        m=m,
        n_orbitals=n_orbitals,
    )

    for orbital_set, expected in zip(result, n_orbitals):
        assert orbital_set.coefficients.shape == (6, expected)


def test_orbital_orthonormality(mock_fem, solver_args):
    """Check C^dagger S C = I in every sector."""

    result = solve_orbitals(
        **solver_args,
        m=np.array([-1, 0, 1]),
        n_orbitals=np.array([2, 3, 2]),
    )

    for orbital_set in result:
        assert orbital_set.check_orthonormality(tolerance=1e-9)
        assert orbital_set.orthonormality_error() < 1e-9


def test_eigenvalue_residuals(mock_fem, solver_args):
    """Check h C - S C epsilon = 0."""

    result = solve_orbitals(
        **solver_args,
        m=np.array([-1, 0, 1]),
        n_orbitals=np.array([2, 3, 2]),
    )

    for orbital_set in result:
        residuals = orbital_set.eigenvalue_residuals()

        assert residuals.shape == (
            orbital_set.n_fem,
            orbital_set.n_orbitals,
        )

        assert np.allclose(residuals, 0.0, atol=1e-9)


def test_residual_norms(mock_fem, solver_args):
    """Stored residual norms must match calculated residuals."""

    result = solve_orbitals(
        **solver_args,
        m=np.array([0, 1]),
        n_orbitals=np.array([2, 2]),
    )

    for orbital_set in result:
        actual = np.linalg.norm(
            orbital_set.eigenvalue_residuals(),
            axis=0,
        )

        assert np.allclose(
            orbital_set.residual_norms,
            actual,
            atol=1e-12,
        )


def test_eigenvalues_are_sorted(mock_fem, solver_args):
    """Orbital energies must be in ascending order."""

    result = solve_orbitals(
        **solver_args,
        m=np.array([-1, 0, 1]),
        n_orbitals=np.array([3, 3, 3]),
    )

    for orbital_set in result:
        assert np.all(np.diff(orbital_set.energies) >= 0.0)


def test_must_be_one_dimensional():
    with pytest.raises(ValueError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=2.0,
            B=0.0,
            m=np.array([[0, 1]]),
            n_orbitals=np.array([2, 2]),
        )


def test_m_and_n_orbitals_must_have_same_length():
    with pytest.raises(ValueError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=2.0,
            B=0.0,
            m=np.array([-1, 0, 1]),
            n_orbitals=np.array([2, 2]),
        )


def test_empty_m_array():
    with pytest.raises(ValueError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=2.0,
            B=0.0,
            m=np.array([], dtype=int),
            n_orbitals=np.array([], dtype=int),
        )


def test_must_contain_integers():
    with pytest.raises(TypeError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=2.0,
            B=0.0,
            m=np.array([-1.0, 0.0, 1.0]),
            n_orbitals=np.array([2, 2, 2]),
        )


def test_n_orbitals_must_contain_integers():
    with pytest.raises(TypeError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=2.0,
            B=0.0,
            m=np.array([-1, 0, 1]),
            n_orbitals=np.array([2.0, 2.0, 2.0]),
        )


def test_n_orbitals_must_be_positive():
    with pytest.raises(ValueError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=2.0,
            B=0.0,
            m=np.array([-1, 0, 1]),
            n_orbitals=np.array([2, 0, 2]),
        )


def test_must_be_unique():
    with pytest.raises(ValueError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=2.0,
            B=0.0,
            m=np.array([-1, 0, 0]),
            n_orbitals=np.array([2, 2, 2]),
        )


def test_invalid_nuclear_charge():
    with pytest.raises(ValueError):
        solve_orbitals(
            rho_nodes=np.array([0.0, 1.0]),
            z_nodes=np.array([-1.0, 1.0]),
            Z=0.0,
            B=0.0,
            m=np.array([0]),
            n_orbitals=np.array([2]),
        )


def test_requested_orbitals_exceed_fem_basis(mock_fem, solver_args):
    with pytest.raises(ValueError):
        solve_orbitals(
            **solver_args,
            m=np.array([0]),
            n_orbitals=np.array([7]),
        )
