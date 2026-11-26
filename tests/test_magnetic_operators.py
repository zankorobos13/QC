import numpy as np
import pytest

from scipy.sparse.linalg import norm as sparse_norm

from grid.mesh import make_atomic_mesh
from operators.fem import assemble_fem_matrices

from numpy.polynomial.legendre import leggauss
from operators.fem import shape_functions


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


@pytest.mark.parametrize("m", [1, 2, 3])
def test_kinetic_matrix_is_even_in_m(matrices, m):
    """
    The kinetic operator depends on m through m^2.

    Therefore:

        T(m) = T(-m)
    """

    T_plus, _, _ = matrices(0.7, m, 0.0)
    T_minus, _, _ = matrices(0.7, -m, 0.0)

    assert_sparse_matrices_close(T_plus, T_minus)

# ============================================================
# 6. ABSOLUTE DIAMAGNETIC CONTRIBUTION
# ============================================================

def assemble_rho_squared_matrix(rho_nodes, z_nodes):
    """
    Independently assemble:

        R2_mu_nu = integral rho^3 * N_mu * N_nu drho dz

    This is the FEM matrix corresponding to multiplication
    by rho^2 in the cylindrical volume element rho drho dz.
    """

    n_rho = len(rho_nodes)
    n_z = len(z_nodes)

    n_total = n_rho * n_z

    R2 = np.zeros((n_total, n_total), dtype=np.float64)

    points, weights = leggauss(QUADRATURE_ORDER)

    for i in range(n_rho - 1):
        for j in range(n_z - 1):

            rho_left = rho_nodes[i]
            rho_right = rho_nodes[i + 1]

            z_bottom = z_nodes[j]
            z_top = z_nodes[j + 1]

            J = (
                (rho_right - rho_left)
                * (z_top - z_bottom)
                / 4.0
            )

            indices = np.array([
                i * n_z + j,
                (i + 1) * n_z + j,
                i * n_z + (j + 1),
                (i + 1) * n_z + (j + 1),
            ])

            local = np.zeros((4, 4), dtype=np.float64)

            for a, xi in enumerate(points):
                for b, eta in enumerate(points):

                    N, _, _ = shape_functions(xi, eta)

                    rho = (
                        (1.0 - xi) * rho_left
                        + (1.0 + xi) * rho_right
                    ) / 2.0

                    weight = (
                        weights[a]
                        * weights[b]
                        * J
                        * rho**3
                    )

                    local += (
                        weight
                        * np.outer(N, N)
                    )

            R2[np.ix_(indices, indices)] += local

    return R2


def test_absolute_diamagnetic_contribution(mesh, matrices):
    """
    Verify the absolute coefficient:

        V(B,m,s) - V(0,m,0)
        - (-B*m/2 + B*s)*S
        = B^2/8 * R2
    """

    rho_nodes, z_nodes = mesh

    B = 0.7
    m = 1
    spin = +0.5

    _, S, V_B = matrices(B, m, spin)
    _, _, V_0 = matrices(0.0, m, 0.0)

    # Obtain the exact FEM free-node indexing.
    _, _, _, free_indices = assemble_fem_matrices(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        Z=Z,
        B=B,
        m=m,
        spin=spin,
        quadrature_order=QUADRATURE_ORDER,
    )

    # Independently assemble the full rho^2 matrix.
    R2_full = assemble_rho_squared_matrix(
        rho_nodes,
        z_nodes,
    )

    # Apply the same Dirichlet reduction as in the Hamiltonian.
    R2 = R2_full[np.ix_(free_indices, free_indices)]

    expected = (B**2 / 8.0) * R2

    zeeman_coefficient = -B * m / 2.0 + B * spin

    actual = (
        V_B
        - V_0
        - zeeman_coefficient * S
    )

    difference = actual.toarray() - expected

    error = np.linalg.norm(difference)
    reference = np.linalg.norm(expected)

    assert error <= MATRIX_ATOL + MATRIX_RTOL * reference, (
        "Absolute diamagnetic contribution mismatch: "
        f"error={error:.6e}, "
        f"reference={reference:.6e}, "
        f"relative_error={error / max(reference, 1e-300):.6e}"
    )