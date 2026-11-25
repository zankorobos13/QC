
from __future__ import annotations

import time
import numpy as np

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals
from orbitals.basis import build_orbital_basis
from integrals.tensor import build_coulomb_tensor
from scf.system import ElectronicSystem
from scf.uhf import run_uhf


# =============================================================
# Physical parameters
# =============================================================

Z = 4.0
B = 0.0

N_ALPHA = 2
N_BETA = 1

M_VALUES = np.array([0], dtype=int)
N_ORBITALS = np.array([16], dtype=int)

QUADRATURE_ORDER = 8

# Mesh mapping
A_RHO = 1.0
A_Z = 1.0
MAPPING_B = 0.5

T_RHO_MAX = 0.98
T_Z_MAX = 0.98

# SCF parameters
MAX_ITERATIONS = 100
ENERGY_TOLERANCE = 1.0e-10
DENSITY_TOLERANCE = 1.0e-8
DAMPING = 0.0

# Ivanov-Schmelcher Be+ UHF reference at B=0.
REFERENCE_ENERGY = -14.27747

# Grid sizes: number of nodes in each direction.
GRID_SIZES = (40, 60, 80, 120)


# =============================================================
# Single-grid calculation
# =============================================================

def run_grid(n_rho: int, n_z: int) -> dict:

    start_total = time.perf_counter()

    print()
    print("=" * 72)
    print(f"Be+ UHF: grid {n_rho} x {n_z}")
    print("=" * 72)

    # ---------------------------------------------------------
    # 1. Build FEM mesh
    # ---------------------------------------------------------

    start = time.perf_counter()

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n_rho,
        n_z=n_z,
        A_rho=A_RHO,
        A_z=A_Z,
        b=MAPPING_B,
        t_rho_max=T_RHO_MAX,
        t_z_max=T_Z_MAX,
    )

    rho_nodes = np.asarray(rho_mesh.physical, dtype=float)
    z_nodes = np.asarray(z_mesh.physical, dtype=float)

    print(
        f"Mesh construction:       "
        f"{time.perf_counter() - start:.3f} s"
    )

    # ---------------------------------------------------------
    # 2. Solve one-electron orbitals
    # ---------------------------------------------------------

    start = time.perf_counter()

    orbital_sets = solve_orbitals(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        Z=Z,
        B=B,
        m=M_VALUES,
        n_orbitals=N_ORBITALS,
        quadrature_order=QUADRATURE_ORDER,
    )

    basis = build_orbital_basis(orbital_sets)

    print(
        f"Orbital calculation:     "
        f"{time.perf_counter() - start:.3f} s"
    )

    print(f"Number of orbitals:      {basis.n_orbitals}")
    print(
        f"Orthonormality error:    "
        f"{basis.check_orthonormality()}"
    )

    # ---------------------------------------------------------
    # 3. Build Coulomb tensor
    # ---------------------------------------------------------

    start = time.perf_counter()

    psi_arr, m_arr = basis.get_orbitals_psi(
        rho_nodes,
        z_nodes,
    )

    g = build_coulomb_tensor(
        basis=basis,
        psi_arr=psi_arr,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        quadrature_order=QUADRATURE_ORDER,
    )

    print(
        f"Coulomb tensor:          "
        f"{time.perf_counter() - start:.3f} s"
    )

    print(f"Tensor shape:            {g.shape}")

    # ---------------------------------------------------------
    # 4. Run UHF-SCF
    # ---------------------------------------------------------

    system = ElectronicSystem(
        Z=Z,
        n_alpha=N_ALPHA,
        n_beta=N_BETA,
        B=B,
    )

    start = time.perf_counter()

    result = run_uhf(
        system=system,
        basis=basis,
        g=g,
        max_iterations=MAX_ITERATIONS,
        energy_tolerance=ENERGY_TOLERANCE,
        density_tolerance=DENSITY_TOLERANCE,
        damping=DAMPING,
    )

    scf_time = time.perf_counter() - start

    total_time = time.perf_counter() - start_total

    energy = float(result.energy)
    error = energy - REFERENCE_ENERGY

    print()
    print("UHF RESULTS")
    print("-" * 72)
    print(f"Converged:               {result.converged}")
    print(f"Iterations:              {result.iterations}")
    print(f"Density error:           {result.density_error:.6e}")
    print(f"Energy:                  {energy:.10f} Ha")
    print(f"Reference energy:        {REFERENCE_ENERGY:.10f} Ha")
    print(f"Energy difference:       {error:+.10f} Ha")
    print(f"SCF time:                {scf_time:.3f} s")
    print(f"Total time:              {total_time:.3f} s")

    return {
        "grid": f"{n_rho}x{n_z}",
        "energy": energy,
        "error": error,
        "converged": result.converged,
        "iterations": result.iterations,
        "density_error": result.density_error,
        "scf_time": scf_time,
        "total_time": total_time,
    }


# =============================================================
# Main
# =============================================================

def main():

    print("=" * 100)
    print("Be+ UHF GRID-CONVERGENCE STUDY")
    print("=" * 100)

    print(f"Z = {Z}")
    print(f"B = {B}")
    print(f"N_alpha = {N_ALPHA}")
    print(f"N_beta = {N_BETA}")
    print(f"m sectors = {M_VALUES}")
    print(f"Orbitals per sector = {N_ORBITALS}")
    print(f"Quadrature order = {QUADRATURE_ORDER}")
    print(f"Reference UHF energy = {REFERENCE_ENERGY:.8f} Ha")

    results = []

    for n in GRID_SIZES:
        result = run_grid(n, n)
        results.append(result)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("GRID-CONVERGENCE SUMMARY")
    print("=" * 100)

    print(
        f"{'Grid':>10}"
        f"{'Energy (Ha)':>20}"
        f"{'Delta E (mHa)':>18}"
        f"{'Iterations':>12}"
        f"{'Converged':>12}"
        f"{'Total time (s)':>18}"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{result['grid']:>10}"
            f"{result['energy']:20.10f}"
            f"{result['error'] * 1000:18.6f}"
            f"{result['iterations']:12d}"
            f"{str(result['converged']):>12}"
            f"{result['total_time']:18.3f}"
        )

    print("=" * 100)

    # ---------------------------------------------------------
    # Energy differences between consecutive grids
    # ---------------------------------------------------------

    print()
    print("ENERGY CHANGES BETWEEN CONSECUTIVE GRIDS")
    print("-" * 72)

    for previous, current in zip(results[:-1], results[1:]):

        delta = current["energy"] - previous["energy"]

        print(
            f"{previous['grid']:>10} -> {current['grid']:<10} "
            f"{delta * 1000:+.6f} mHa"
        )

    print("-" * 72)

    if not all(r["converged"] for r in results):
        print("WARNING: At least one SCF calculation did not converge.")


if __name__ == "__main__":
    main()
