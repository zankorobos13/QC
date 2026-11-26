
import numpy as np

from grid.mesh import make_atomic_mesh

from orbitals.solver import solve_orbitals
from orbitals.basis import build_orbital_basis

from integrals.tensor import build_coulomb_tensor

from scf.system import ElectronicSystem
from scf.uhf import run_uhf

import time


def main():

    # ---------------------------------------------------------
    # 1. Physical system: helium atom
    # ---------------------------------------------------------

    Z = 4.0
    B = 5000


    n_alpha = 1
    n_beta = 2

    system = ElectronicSystem(
        Z=Z,
        n_alpha=n_alpha,
        n_beta=n_beta,
        B=B,
    )

    # ---------------------------------------------------------
    # 2. FEM grid
    # ---------------------------------------------------------

    n_rho = 80
    n_z = 80

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n_rho,
        n_z=n_z,
        A_rho=1.0,
        A_z=1.0,
        b=0.5,
        t_rho_max=0.98,
        t_z_max=0.98
    )

    rho_nodes = rho_mesh.physical
    z_nodes = z_mesh.physical

    # ---------------------------------------------------------
    # 3. Spatial orbital calculation
    # ---------------------------------------------------------

    m_values = np.array([0], dtype=int)
    orbital_counts = np.array([16], dtype=int)

    print("=" * 65)
    print("Be+ UHF-SCF CALCULATION")
    print("=" * 65)

    print("\nBuilding spatial orbitals...")

    orbital_sets = solve_orbitals(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        Z=Z,
        B=B,
        m=m_values,
        n_orbitals=orbital_counts,
        quadrature_order=8,
    )

    # ---------------------------------------------------------
    # 4. Assemble global orbital basis
    # ---------------------------------------------------------

    basis = build_orbital_basis(orbital_sets)

    print(f"Number of spatial orbitals: {basis.n_orbitals}")
    print(
        "Basis orthonormality error:",
        basis.orthonormality_errors(),
    )

    # ---------------------------------------------------------
    # 5. Two-electron Coulomb integrals
    # ---------------------------------------------------------

    print("\nBuilding Coulomb tensor...")

    psi_arr, m_arr = basis.get_orbitals_psi(
        rho_nodes,
        z_nodes,
    )

    t = time.time()

    g = build_coulomb_tensor(
        basis=basis,
        psi_arr=psi_arr,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        quadrature_order=8,
    )

    print("Time elapsed: ", time.time() - t)

    print("Coulomb tensor shape:", g.shape)

    # ---------------------------------------------------------
    # 6. UHF-SCF
    # ---------------------------------------------------------

    print("\nRunning UHF-SCF...")

    result = run_uhf(
        system=system,
        basis=basis,
        g=g,
        max_iterations=300,
        energy_tolerance=1e-13,
        density_tolerance=1e-12,
        damping=0.0,
    )

    # ---------------------------------------------------------
    # 7. Results
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("UHF RESULTS")
    print("=" * 65)

    print(f"Converged:              {result.converged}")
    print(f"Iterations:             {result.iterations}")
    print(f"Electronic energy:      {result.energy:.12f} Ha")
    print(f"Density error:          {result.density_error:.3e}")

    print("\nAlpha orbital energies:")
    print(result.orbital_energies_alpha)

    print("\nBeta orbital energies:")
    print(result.orbital_energies_beta)

    # ---------------------------------------------------------
    # 8. Density matrix checks
    # ---------------------------------------------------------

    D_alpha = result.density_alpha
    D_beta = result.density_beta

    D_total = D_alpha + D_beta

    print("\n" + "=" * 65)
    print("DENSITY MATRIX CHECKS")
    print("=" * 65)

    print("Tr(D_alpha):", np.trace(D_alpha).real)
    print("Tr(D_beta): ", np.trace(D_beta).real)
    print("Tr(D_total):", np.trace(D_total).real)

    print(
        "Alpha Hermiticity error:",
        np.linalg.norm(D_alpha - D_alpha.conj().T),
    )

    print(
        "Beta Hermiticity error:",
        np.linalg.norm(D_beta - D_beta.conj().T),
    )

    print(
        "Alpha idempotency error:",
        np.linalg.norm(D_alpha @ D_alpha - D_alpha),
    )

    print(
        "Beta idempotency error:",
        np.linalg.norm(D_beta @ D_beta - D_beta),
    )

    # ---------------------------------------------------------
    # 9. UHF spin checks
    # ---------------------------------------------------------

    S2 = (
        (n_alpha - n_beta) ** 2 / 4
        + (n_alpha + n_beta) / 2
        - np.trace(D_alpha @ D_beta).real
    )

    print("\n" + "=" * 65)
    print("SPIN CHECKS")
    print("=" * 65)

    print(f"<S^2>: {S2:.12f}")

    # ---------------------------------------------------------
    # 10. Reference comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("REFERENCE COMPARISON")
    print("=" * 65)
    
    print(f"Calculated energy:                    {result.energy:.12f} Ha")

    print("\nCalculation finished.")


if __name__ == "__main__":
    main()
