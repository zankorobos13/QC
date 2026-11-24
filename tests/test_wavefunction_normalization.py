import numpy as np
import pytest

from numpy.polynomial.legendre import leggauss

from grid.mesh import make_atomic_mesh
from operators.fem import shape_functions
from orbitals import solve_orbitals


def integrate_wavefunction_norm(
    orbitals,
    rho_nodes,
    z_nodes,
    quadrature_order=10,
    orbital_index=0,
):
    """
    Numerically calculate:

        integral |psi_m(rho,z)|^2 * rho drho dz

    The integration is performed element by element
    using Gauss-Legendre quadrature.
    """

    nr = len(rho_nodes)
    nz = len(z_nodes)

    # Restore the full coefficient grid.
    # Dirichlet boundary values are zero.
    full_coefficients = np.zeros(
        (nr * nz, orbitals.n_orbitals),
        dtype=orbitals.coefficients.dtype,
    )

    full_coefficients[orbitals.free_indices, :] = (
        orbitals.coefficients
    )

    full_coefficients = full_coefficients.reshape(
        nr, nz, orbitals.n_orbitals
    )

    # Gauss-Legendre quadrature points and weights.
    points, weights = leggauss(quadrature_order)

    norm = 0.0

    # Loop over all rectangular FEM elements.
    for i in range(nr - 1):
        rho_left = rho_nodes[i]
        rho_right = rho_nodes[i + 1]

        h_rho = rho_right - rho_left

        for j in range(nz - 1):
            z_bottom = z_nodes[j]
            z_top = z_nodes[j + 1]

            h_z = z_top - z_bottom

            # Local nodal coefficients:
            # (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            local_coefficients = np.array([
                full_coefficients[i, j, orbital_index],
                full_coefficients[i + 1, j, orbital_index],
                full_coefficients[i, j + 1, orbital_index],
                full_coefficients[i + 1, j + 1, orbital_index],
            ])

            # Integrate over the reference element [-1,1]^2.
            for a, xi in enumerate(points):
                for b, eta in enumerate(points):

                    N, _, _ = shape_functions(xi, eta)

                    # Physical coordinates.
                    rho = (
                        rho_left
                        + 0.5 * (xi + 1.0) * h_rho
                    )

                    # Reconstructed FEM orbital.
                    psi = np.dot(N, local_coefficients)

                    # Jacobian of the reference-to-physical map.
                    jacobian = 0.25 * h_rho * h_z

                    norm += (
                        weights[a]
                        * weights[b]
                        * jacobian
                        * rho
                        * abs(psi)**2
                    )

    return float(np.real(norm))


@pytest.mark.parametrize("m", [0, 1, -1])
def test_wavefunction_normalization(m):

    # ---------------------------------------------------------
    # 1. Build FEM mesh
    # ---------------------------------------------------------

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=40,
        n_z=40,
        A_rho=1.0,
        A_z=1.0,
        b=0.5,
        t_rho_max=0.98,
        t_z_max=0.98,
    )

    rho_nodes = rho_mesh.physical
    z_nodes = z_mesh.physical

    # ---------------------------------------------------------
    # 2. Solve the one-electron problem
    # ---------------------------------------------------------

    orbitals_sets = solve_orbitals(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        Z=1,
        B=0.0,
        m=np.array([m]),
        n_orbitals=np.array([1]),
    )

    orbitals = orbitals_sets[0]

    # ---------------------------------------------------------
    # 3. Independently integrate the orbital norm
    # ---------------------------------------------------------

    norm = integrate_wavefunction_norm(
        orbitals=orbitals,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        quadrature_order=10,
        orbital_index=0,
    )

    print(f"\nm = {m}")
    print(f"Energy = {orbitals.energies[0]:.12f} Eh")
    print(f"Wavefunction norm = {norm:.12f}")
    print(f"Normalization error = {abs(norm - 1.0):.3e}")

    # ---------------------------------------------------------
    # 4. Check normalization
    # ---------------------------------------------------------

    assert np.isclose(norm, 1.0, atol=1e-9, rtol=0.0)
