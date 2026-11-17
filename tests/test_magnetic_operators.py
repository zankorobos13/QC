import numpy as np
import pytest

from scipy.sparse.linalg import norm as sparse_norm

from grid.mesh import make_atomic_mesh
from operators.fem import assemble_fem_matrices


# ============================================================
# TEST CONFIGURATION
# ============================================================

Z = 1

N_RHO = 20
N_Z = 20

A_RHO = 1.0
A_Z = 1.0

B_POISSON = 0.5

T_RHO_MAX = 0.98
T_Z_MAX = 0.98

QUADRATURE_ORDER = 4

MATRIX_ATOL = 1e-10
MATRIX_RTOL = 1e-10


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def mesh():
    """
    Construct a relatively small mesh for operator tests.
    """

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=N_RHO,
        n_z=N_Z,
        A_rho=A_RHO,
        A_z=A_Z,
        b=B_POISSON,
        t_rho_max=T_RHO_MAX,
        t_z_max=T_Z_MAX,
    )

    return rho_mesh.physical, z_mesh.physical


@pytest.fixture(scope="module")
def matrices(mesh):
    """
    Cache FEM matrices for all requested combinations
    of B, m and spin.
    """

    rho_nodes, z_nodes = mesh

    cache = {}

    def get(B, m, spin=0.0):

        key = (float(B), int(m), float(spin))

        if key not in cache:

            T, S, V, free_indices = assemble_fem_matrices(
                rho_nodes=rho_nodes,
                z_nodes=z_nodes,
                Z=Z,
                B=B,
                m=m,
                spin=spin,
                quadrature_order=QUADRATURE_ORDER,
            )

            cache[key] = (T, S, V)

        return cache[key]

    return get


# ============================================================
# ASSERTION UTILITIES
# ============================================================

def assert_sparse_matrices_close(actual, expected):
    """
    Compare sparse matrices using Frobenius norm.
    """

    difference = sparse_norm(actual - expected)
    reference = sparse_norm(expected)

    assert difference <= MATRIX_ATOL + MATRIX_RTOL * reference, (
        f"Matrix mismatch: "
        f"error={difference:.6e}, "
        f"reference={reference:.6e}"
    )


# ============================================================
# 1. ORBITAL ZEEMAN TERM
# ============================================================

@pytest.mark.parametrize("m", [-1, 0, 1])
@pytest.mark.parametrize("B", [0.2, 0.7])
def test_orbital_zeeman_term(matrices, B, m):
    """
    V_orb = -B*m/2 * S

    Therefore:

        V(B,m,0) - V(-B,m,0) = -B*m*S
    """

    _, S, V_plus = matrices(B, m, 0.0)
    _, _, V_minus = matrices(-B, m, 0.0)

    expected = -B * m * S

    actual = V_plus - V_minus

    assert_sparse_matrices_close(actual, expected)


# ============================================================
# 2. DIAMAGNETIC TERM
# ============================================================

@pytest.mark.parametrize("m", [-1, 0, 1])
@pytest.mark.parametrize("B", [0.2, 0.7])
def test_diamagnetic_quadratic_dependence(matrices, B, m):
    """
    V_dia = B^2 * rho^2 / 8

    The diamagnetic contribution is even in B.

    Define:

        D(B) = [V(B) + V(-B)]/2 - V(0)

    Then:

        D(2B) = 4*D(B)
    """

    _, _, V_0 = matrices(0.0, m, 0.0)

    _, _, V_plus = matrices(B, m, 0.0)
    _, _, V_minus = matrices(-B, m, 0.0)

    _, _, V_plus_2 = matrices(2.0 * B, m, 0.0)
    _, _, V_minus_2 = matrices(-2.0 * B, m, 0.0)

    D_B = 0.5 * (V_plus + V_minus) - V_0

    D_2B = 0.5 * (V_plus_2 + V_minus_2) - V_0

    assert_sparse_matrices_close(D_2B, 4.0 * D_B)


# ============================================================
# 3. SPIN ZEEMAN TERM
# ============================================================

@pytest.mark.parametrize("m", [-1, 0, 1])
@pytest.mark.parametrize("B", [0.2, 0.7])
def test_spin_zeeman_term(matrices, B, m):
    """
    V_spin = B*spin*S

    For spin = +1/2:

        V_up - V_0 = +B/2*S

    For spin = -1/2:

        V_down - V_0 = -B/2*S
    """

    _, S, V_0 = matrices(B, m, 0.0)

    _, _, V_up = matrices(B, m, +0.5)
    _, _, V_down = matrices(B, m, -0.5)

    assert_sparse_matrices_close(
        V_up - V_0,
        +0.5 * B * S,
    )

    assert_sparse_matrices_close(
        V_down - V_0,
        -0.5 * B * S,
    )


# ============================================================
# 4. SPIN SPLITTING
# ============================================================

@pytest.mark.parametrize("m", [-1, 0, 1])
@pytest.mark.parametrize("B", [0.2, 0.7])
def test_spin_splitting(matrices, B, m):
    """
    V_up - V_down = B*S

    Consequently, for the same spatial orbital:

        E_up - E_down = B
    """

    _, S, V_up = matrices(B, m, +0.5)
    _, _, V_down = matrices(B, m, -0.5)

    assert_sparse_matrices_close(
        V_up - V_down,
        B * S,
    )


# ============================================================
# 5. ZERO-FIELD LIMIT
# ============================================================

@pytest.mark.parametrize("m", [-1, 0, 1])
def test_zero_field_independence_from_spin(matrices, m):
    """
    At B=0, both Zeeman contributions vanish.

    Therefore:

        V(0,m,+1/2) = V(0,m,-1/2) = V(0,m,0)
    """

    _, _, V_0 = matrices(0.0, m, 0.0)
    _, _, V_up = matrices(0.0, m, +0.5)
    _, _, V_down = matrices(0.0, m, -0.5)

    assert_sparse_matrices_close(V_up, V_0)
    assert_sparse_matrices_close(V_down, V_0)
