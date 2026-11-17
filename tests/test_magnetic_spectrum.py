
import numpy as np
import pytest

from scipy.sparse.linalg import eigsh

from grid.mesh import make_atomic_mesh
from operators.fem import assemble_fem_matrices


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

Z = 1

N_RHO = 30
N_Z = 30

A_RHO = 1.0
A_Z = 1.0
B_POISSON = 0.5

T_MAX = 0.98
QUADRATURE_ORDER = 4

N_STATES = 3

ENERGY_ATOL = 2e-7


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def mesh():
    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=N_RHO,
        n_z=N_Z,
        A_rho=A_RHO,
        A_z=A_Z,
        b=B_POISSON,
        t_rho_max=T_MAX,
        t_z_max=T_MAX,
    )

    return rho_mesh.physical, z_mesh.physical


@pytest.fixture(scope="module")
def spectrum_solver(mesh):

    rho_nodes, z_nodes = mesh
    cache = {}

    def solve(B, m, spin=0.0, n_states=N_STATES):

        key = (float(B), int(m), float(spin), int(n_states))

        if key in cache:
            return cache[key]

        T, S, V, free_indices = assemble_fem_matrices(
            rho_nodes=rho_nodes,
            z_nodes=z_nodes,
            Z=Z,
            B=B,
            m=m,
            spin=spin,
            quadrature_order=QUADRATURE_ORDER,
        )

        H = T + V

        # Approximate ground-state energy used as a shift.
        sigma = -0.5 - 0.5 * B * m + B * spin

        energies, _ = eigsh(
            H,
            k=n_states,
            M=S,
            sigma=sigma,
            which="LM",
            tol=1e-9,
            maxiter=100000,
        )

        energies = np.sort(energies)

        cache[key] = energies
        return energies

    return solve


# ---------------------------------------------------------
# Spin Zeeman splitting
# ---------------------------------------------------------

@pytest.mark.parametrize("B", [0.2, 0.5])
@pytest.mark.parametrize("m", [-1, 0, 1])
def test_spectral_spin_splitting(spectrum_solver, B, m):

    E_up = spectrum_solver(B, m, spin=+0.5)
    E_down = spectrum_solver(B, m, spin=-0.5)

    # H(+1/2) = H(0) + B/2 S
    # H(-1/2) = H(0) - B/2 S
    #
    # Therefore the energy difference is exactly B.

    splitting = E_up - E_down

    np.testing.assert_allclose(
        splitting,
        B,
        atol=ENERGY_ATOL,
        rtol=0.0,
    )


# ---------------------------------------------------------
# Orbital Zeeman splitting
# ---------------------------------------------------------

@pytest.mark.parametrize("B", [0.2, 0.5])
def test_spectral_orbital_splitting(spectrum_solver, B):

    E_m_plus = spectrum_solver(B, m=+1, spin=0.0)
    E_m_minus = spectrum_solver(B, m=-1, spin=0.0)

    # The m=+1 and m=-1 Hamiltonians differ
    # only by the orbital Zeeman contribution.
    #
    # E(+1) - E(-1) = -B.

    np.testing.assert_allclose(
        E_m_plus - E_m_minus,
        -B,
        atol=ENERGY_ATOL,
        rtol=0.0,
    )


# ---------------------------------------------------------
# Degeneracy at zero magnetic field
# ---------------------------------------------------------

def test_spectral_m_degeneracy_at_zero_field(spectrum_solver):

    E_plus = spectrum_solver(0.0, m=+1, spin=0.0)
    E_minus = spectrum_solver(0.0, m=-1, spin=0.0)

    np.testing.assert_allclose(
        E_plus,
        E_minus,
        atol=ENERGY_ATOL,
        rtol=0.0,
    )


# ---------------------------------------------------------
# Diamagnetic energy shift
# ---------------------------------------------------------

def test_diamagnetic_energy_is_monotonic(spectrum_solver):

    fields = [0.0, 0.2, 0.4, 0.6]

    ground_energies = []

    for B in fields:
        energy = spectrum_solver(
            B=B,
            m=0,
            spin=0.0,
            n_states=1,
        )[0]

        ground_energies.append(energy)

    ground_energies = np.asarray(ground_energies)

    # For m=0:
    # H(B) = H(0) + B^2 rho^2 / 8.
    #
    # The added operator is positive semidefinite.
    # Therefore the ground-state energy cannot decrease.

    differences = np.diff(ground_energies)

    assert np.all(differences >= -ENERGY_ATOL), (
        f"Ground-state energies are not monotonic: "
        f"{ground_energies}"
    )
