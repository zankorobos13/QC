import numpy as np

from grid.mesh import make_atomic_mesh
from orbitals import (
    solve_orbitals,
    evaluate_orbital,
    evaluate_orbital_grid,
    evaluate_wavefunction,
)


def test_orbital_evaluation():
    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=30,
        n_z=30,
        A_rho=1.0,
        A_z=1.0,
        b=0.5,
        t_rho_max=0.98,
        t_z_max=0.98,
    )

    orbitals = solve_orbitals(
        rho_nodes=rho_mesh.physical,
        z_nodes=z_mesh.physical,
        Z=1,
        B=0.0,
        m=0,
        n_orbitals=3,
    )

    # 1. Reconstruct values on the complete grid.
    psi_grid = evaluate_orbital_grid(
        orbitals,
        rho_mesh.physical,
        z_mesh.physical,
        orbital_index=0,
    )

    assert psi_grid.shape == (
        len(rho_mesh.physical),
        len(z_mesh.physical),
    )

    # 2. Check interpolation at an interior mesh node.
    i, j = 10, 15

    psi_point = evaluate_orbital(
        orbitals,
        rho_mesh.physical,
        z_mesh.physical,
        rho_mesh.physical[i],
        z_mesh.physical[j],
        orbital_index=0,
    )

    assert np.isclose(psi_point, psi_grid[i, j])

    # 3. Check full 3D wavefunction for m = 0.
    psi_3d = evaluate_wavefunction(
        orbitals,
        rho_mesh.physical,
        z_mesh.physical,
        rho=0.5,
        phi=0.7,
        z=0.0,
    )

    psi_2d = evaluate_orbital(
        orbitals,
        rho_mesh.physical,
        z_mesh.physical,
        rho=0.5,
        z=0.0,
    )

    assert np.isclose(
        psi_3d,
        psi_2d / np.sqrt(2.0 * np.pi),
    )