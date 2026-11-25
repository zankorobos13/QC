import time
from dataclasses import replace

import numpy as np

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals
from orbitals.basis import build_orbital_basis
from integrals.tensor import build_coulomb_tensor
from scf.system import ElectronicSystem
from scf.uhf import run_uhf


# =============================================================
# Parameters
# =============================================================

Z = 4.0
B = 0.0

N_ALPHA = 2
N_BETA = 1

M_VALUES = np.array([0], dtype=int)

MAX_ORBITALS = 32
ORBITAL_COUNTS = (20, 24, 32)

N_RHO = 80
N_Z = 80

A_RHO = 1.0
A_Z = 1.0
MESH_B = 0.5
T_RHO_MAX = 0.98
T_Z_MAX = 0.98

QUADRATURE_ORDER = 8

REFERENCE_ENERGY = -14.27747000

MAX_ITERATIONS = 100
ENERGY_TOLERANCE = 1e-10
DENSITY_TOLERANCE = 1e-8
DAMPING = 0.0


# =============================================================
# Mesh
# =============================================================

def build_mesh():
    rho_mesh, z_mesh = make_atomic_mesh(
        N_RHO,
        N_Z,
        A_RHO,
        A_Z,
        MESH_B,
        T_RHO_MAX,
        T_Z_MAX,
    )

    return rho_mesh.physical, z_mesh.physical


# =============================================================
# Main calculation
# =============================================================

def main():

    print("=" * 90)
    print("Be+ UHF ORBITAL-BASIS CONVERGENCE")
    print("=" * 90)

    print(f"Z = {Z}")
    print(f"B = {B}")
    print(f"N_alpha = {N_ALPHA}")
    print(f"N_beta = {N_BETA}")
    print(f"m sectors = {M_VALUES.tolist()}")
    print(f"Grid = {N_RHO} x {N_Z}")
    print(f"Quadrature order = {QUADRATURE_ORDER}")
    print(f"Orbital counts = {ORBITAL_COUNTS}")
    print(f"Reference UHF energy = {REFERENCE_ENERGY:.8f} Ha")

    # ---------------------------------------------------------
    # Mesh
    # ---------------------------------------------------------

    start = time.perf_counter()

    rho_nodes, z_nodes = build_mesh()

    print(f"\nMesh construction: {time.perf_counter() - start:.3f} s")

    # ---------------------------------------------------------
    # Calculate maximum orbital basis
    # ---------------------------------------------------------

    print("\nCalculating spatial orbitals...")

    start = time.perf_counter()

    orbital_sets = solve_orbitals(
        rho_nodes,
        z_nodes,
        Z,
        B,
        M_VALUES,
        np.array([MAX_ORBITALS]),
        quadrature_order=QUADRATURE_ORDER,
    )

    print(f"Orbital calculation: {time.perf_counter() - start:.3f} s")

    # ---------------------------------------------------------
    # Full basis
    # ---------------------------------------------------------

    full_basis = build_orbital_basis(orbital_sets)

    print(f"Number of orbitals: {full_basis.n_orbitals}")
    print(f"Orthonormality errors: {full_basis.orthonormality_errors()}")

    # ---------------------------------------------------------
    # Full Coulomb tensor
    # ---------------------------------------------------------

    print("\nBuilding full Coulomb tensor...")

    psi_arr, m_arr = full_basis.get_orbitals_psi(
        rho_nodes,
        z_nodes,
    )

    start = time.perf_counter()

    g_full = build_coulomb_tensor(
        full_basis,
        psi_arr,
        rho_nodes,
        z_nodes,
        quadrature_order=QUADRATURE_ORDER,
    )

    tensor_time = time.perf_counter() - start

    print(f"Coulomb tensor time: {tensor_time:.3f} s")
    print(f"Coulomb tensor shape: {g_full.shape}")

    # ---------------------------------------------------------
    # UHF calculations for truncated bases
    # ---------------------------------------------------------

    system = ElectronicSystem(
        Z,
        N_ALPHA,
        N_BETA,
        B,
    )

    results = []

    for n_orbitals in ORBITAL_COUNTS:

        print("\n" + "=" * 90)
        print(f"UHF calculation: {n_orbitals} orbitals")
        print("=" * 90)

        start = time.perf_counter()

        # Truncate the orbital set.
        truncated_set = replace(
            orbital_sets[0],
            energies=orbital_sets[0].energies[:n_orbitals],
            coefficients=orbital_sets[0].coefficients[:, :n_orbitals],
            residual_norms=orbital_sets[0].residual_norms[:n_orbitals],
        )

        basis = build_orbital_basis([truncated_set])

        # The tensor uses the same orbital ordering.
        g = g_full[
            :n_orbitals,
            :n_orbitals,
            :n_orbitals,
            :n_orbitals,
        ]

        print(f"Basis size: {basis.n_orbitals}")
        print(f"Tensor shape: {g.shape}")
        print(f"Orthonormality errors: {basis.orthonormality_errors()}")

        result = run_uhf(
            system,
            basis,
            g,
            max_iterations=MAX_ITERATIONS,
            energy_tolerance=ENERGY_TOLERANCE,
            density_tolerance=DENSITY_TOLERANCE,
            damping=DAMPING,
        )

        elapsed = time.perf_counter() - start

        energy = result.energy
        delta = energy - REFERENCE_ENERGY

        results.append({
            "n": n_orbitals,
            "energy": energy,
            "delta": delta,
            "iterations": result.iterations,
            "converged": result.converged,
            "density_error": result.density_error,
            "time": elapsed,
        })

        print("\nUHF RESULTS")
        print("-" * 70)
        print(f"Converged:          {result.converged}")
        print(f"Iterations:         {result.iterations}")
        print(f"Density error:      {result.density_error:.6e}")
        print(f"Energy:             {energy:.10f} Ha")
        print(f"Reference energy:   {REFERENCE_ENERGY:.10f} Ha")
        print(f"Energy difference:  {delta:+.10f} Ha")
        print(f"Elapsed time:       {elapsed:.3f} s")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("ORBITAL-BASIS CONVERGENCE SUMMARY")
    print("=" * 100)

    print(
        f"{'Orbitals':>10}"
        f"{'Energy (Ha)':>20}"
        f"{'Delta E (mHa)':>18}"
        f"{'Iterations':>13}"
        f"{'Converged':>12}"
        f"{'Time (s)':>12}"
    )

    print("-" * 100)

    for item in results:
        print(
            f"{item['n']:>10d}"
            f"{item['energy']:>20.10f}"
            f"{item['delta'] * 1000:>18.6f}"
            f"{item['iterations']:>13d}"
            f"{str(item['converged']):>12}"
            f"{item['time']:>12.3f}"
        )

    print("=" * 100)

    print("\nENERGY CHANGES BETWEEN BASIS SIZES")
    print("-" * 70)

    for previous, current in zip(results[:-1], results[1:]):
        difference = current["energy"] - previous["energy"]

        print(
            f"{previous['n']:>3d} -> {current['n']:<3d}"
            f" {difference * 1000:+.6f} mHa"
        )

    print("-" * 70)


if __name__ == "__main__":
    main()
