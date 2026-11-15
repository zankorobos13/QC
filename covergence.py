import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh

from grid.mesh import make_atomic_mesh
from operators.fem import assemble_fem_matrices


def calculate_ground_state(n, quadrature_order=8):
    Z = 1
    B = 0.0
    m = 0
    spin = 0.0

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n,
        n_z=n,
        A_rho=1.0,
        A_z=1.0,
        b=0.5,
        t_rho_max=0.95,
        t_z_max=0.95,
    )

    T, S, V, free_indices = assemble_fem_matrices(
        rho_nodes=rho_mesh.physical,
        z_nodes=z_mesh.physical,
        Z=Z,
        B=B,
        m=m,
        spin=spin,
        quadrature_order=quadrature_order,
    )

    H = (T + V).tocsc()

    eigenvalues = eigsh(
        H,
        k=6,
        M=S,
        sigma=-0.45,
        which="LM",
        tol=1e-10,
        maxiter=200000,
        ncv=40,
    )[0]

    return np.sort(eigenvalues)


def main():
    grid_sizes = [40, 60, 80, 100, 120]

    exact_energy = -0.5

    results = []

    print("HYDROGEN GRID CONVERGENCE")
    print("-" * 65)
    print(
        f"{'Grid':>10}"
        f"{'Energy (Eh)':>20}"
        f"{'Absolute error':>20}"
    )

    for n in grid_sizes:
        energies = calculate_ground_state(n)

        print(f"\nGrid: {n} x {n}")

        for i, energy in enumerate(energies):
            print(f"State {i + 1}: {energy:.12f} Eh")


if __name__ == "__main__":
    main()
