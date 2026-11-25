
import numpy as np

from scf.energy import calculate_electronic_energy
from scf.fock import build_fock_matrices

import time
from dataclasses import replace

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals
from orbitals.basis import build_orbital_basis
from integrals.tensor import build_coulomb_tensor
from scf.system import ElectronicSystem
from scf.uhf import run_uhf

def _matrix_error(A, B):
    return float(np.max(np.abs(A - B)))


def _direct_electronic_energy(
    h_alpha,
    h_beta,
    g,
    D_alpha,
    D_beta,
):
    """
    Independent energy evaluation from the one- and two-electron
    integrals.

    Integral convention:
        g[p,q,r,s] = <p q | r s>

    Density convention:
        D[p,q] = sum_i C[p,i] conj(C[q,i])
    """

    D_total = D_alpha + D_beta

    # One-electron contribution.
    E_one = np.einsum(
        "pq,qp->",
        D_alpha,
        h_alpha,
        optimize=True,
    ) + np.einsum(
        "pq,qp->",
        D_beta,
        h_beta,
        optimize=True,
    )

    # Coulomb contribution:
    #
    # 1/2 sum_pqrs D_total[r,p] D_total[s,q] g[p,q,r,s]
    #
    E_coulomb = 0.5 * np.einsum(
        "rp,sq,pqrs->",
        D_total,
        D_total,
        g,
        optimize=True,
    )

    # Same-spin exchange contribution:
    #
    # -1/2 sum_pqrs D_sigma[s,p] D_sigma[r,q] g[p,q,r,s]
    #
    E_exchange_alpha = -0.5 * np.einsum(
        "sp,rq,pqrs->",
        D_alpha,
        D_alpha,
        g,
        optimize=True,
    )

    E_exchange_beta = -0.5 * np.einsum(
        "sp,rq,pqrs->",
        D_beta,
        D_beta,
        g,
        optimize=True,
    )

    E_two = (
        E_coulomb
        + E_exchange_alpha
        + E_exchange_beta
    )

    return {
        "one_electron": float(np.real(E_one)),
        "coulomb": float(np.real(E_coulomb)),
        "exchange_alpha": float(np.real(E_exchange_alpha)),
        "exchange_beta": float(np.real(E_exchange_beta)),
        "two_electron": float(np.real(E_two)),
        "total": float(np.real(E_one + E_two)),
    }


def _check_density(name, D, expected_electrons):
    trace = np.trace(D)
    idempotency = np.linalg.norm(D @ D - D)
    hermiticity = np.linalg.norm(D - D.conj().T)

    print(f"\n{name} density:")
    print(f"  Trace:              {trace.real:.12f}")
    print(f"  Trace imaginary:    {abs(trace.imag):.3e}")
    print(f"  Expected trace:     {expected_electrons}")
    print(f"  Idempotency error:  {idempotency:.3e}")
    print(f"  Hermiticity error:  {hermiticity:.3e}")

    return {
        "trace_error": abs(trace - expected_electrons),
        "idempotency_error": idempotency,
        "hermiticity_error": hermiticity,
    }


def _check_orbitals(name, C, D, n_occupied):
    C_occ = C[:, :n_occupied]

    orthogonality = np.linalg.norm(
        C_occ.conj().T @ C_occ
        - np.eye(n_occupied)
    )

    reconstructed_density = C_occ @ C_occ.conj().T

    density_error = np.linalg.norm(
        reconstructed_density - D
    )

    print(f"\n{name} orbitals:")
    print(f"  Orthogonality error:       {orthogonality:.3e}")
    print(f"  Density reconstruction:   {density_error:.3e}")

    return {
        "orthogonality_error": orthogonality,
        "density_reconstruction_error": density_error,
    }


def run_uhf_consistency_test(system, basis, g, result):
    """
    Run consistency checks using an existing UHF result.

    No orbital calculation, Poisson solve or tensor construction
    is performed here.
    """

    np.set_printoptions(precision=12, suppress=True)

    n = basis.n_orbitals

    g = np.asarray(g, dtype=np.complex128)

    if g.shape != (n, n, n, n):
        raise ValueError("Invalid Coulomb tensor shape.")

    h_alpha, h_beta = system.get_one_electron_matrices(basis)

    D_alpha = np.asarray(result.density_alpha)
    D_beta = np.asarray(result.density_beta)

    C_alpha = np.asarray(result.coefficients_alpha)
    C_beta = np.asarray(result.coefficients_beta)

    print("=" * 75)
    print("UHF CONSISTENCY TEST")
    print("=" * 75)

    print(f"Number of orbitals: {n}")
    print(f"N_alpha: {system.n_alpha}")
    print(f"N_beta:  {system.n_beta}")
    print(f"SCF energy: {result.energy:.12f} Ha")
    print(f"SCF converged: {result.converged}")
    print(f"SCF iterations: {result.iterations}")

    # ---------------------------------------------------------
    # 1. Density matrices
    # ---------------------------------------------------------

    alpha_density = _check_density(
        "Alpha",
        D_alpha,
        system.n_alpha,
    )

    beta_density = _check_density(
        "Beta",
        D_beta,
        system.n_beta,
    )

    # ---------------------------------------------------------
    # 2. Orbital orthonormality and density reconstruction
    # ---------------------------------------------------------

    alpha_orbitals = _check_orbitals(
        "Alpha",
        C_alpha,
        D_alpha,
        system.n_alpha,
    )

    beta_orbitals = _check_orbitals(
        "Beta",
        C_beta,
        D_beta,
        system.n_beta,
    )

    # ---------------------------------------------------------
    # 3. Fock matrices
    # ---------------------------------------------------------

    (
        F_alpha,
        F_beta,
        J,
        K_alpha,
        K_beta,
    ) = build_fock_matrices(
        h_alpha,
        h_beta,
        g,
        D_alpha,
        D_beta,
    )

    print("\nFock matrix diagnostics:")
    print(
        "  Alpha Hermiticity:",
        f"{np.linalg.norm(F_alpha - F_alpha.conj().T):.3e}",
    )
    print(
        "  Beta Hermiticity: ",
        f"{np.linalg.norm(F_beta - F_beta.conj().T):.3e}",
    )

    print(
        "  Returned alpha Fock difference:",
        f"{_matrix_error(F_alpha, result.fock_alpha):.3e}",
    )
    print(
        "  Returned beta Fock difference: ",
        f"{_matrix_error(F_beta, result.fock_beta):.3e}",
    )

    # ---------------------------------------------------------
    # 4. Alternative Coulomb index ordering
    # ---------------------------------------------------------

    D_total = D_alpha + D_beta

    J_rs = np.einsum(
        "rs,arbs->ab",
        D_total,
        g,
        optimize=True,
    )

    J_sr = np.einsum(
        "rs,asbr->ab",
        D_total,
        g,
        optimize=True,
    )

    print("\nCoulomb index-ordering diagnostics:")
    print(
        "  Difference J(arbs) vs J(asbr):",
        f"{_matrix_error(J_rs, J_sr):.3e}",
    )

    # ---------------------------------------------------------
    # 5. Energy from existing Fock implementation
    # ---------------------------------------------------------

    energy_fock = calculate_electronic_energy(
        h_alpha,
        h_beta,
        F_alpha,
        F_beta,
        D_alpha,
        D_beta,
    )

    # ---------------------------------------------------------
    # 6. Independent energy from g
    # ---------------------------------------------------------

    energy_direct = _direct_electronic_energy(
        h_alpha,
        h_beta,
        g,
        D_alpha,
        D_beta,
    )

    print("\nEnergy decomposition:")
    print(f"  One-electron:       {energy_direct['one_electron']:.12f}")
    print(f"  Coulomb:            {energy_direct['coulomb']:.12f}")
    print(f"  Alpha exchange:     {energy_direct['exchange_alpha']:.12f}")
    print(f"  Beta exchange:      {energy_direct['exchange_beta']:.12f}")
    print(f"  Two-electron:       {energy_direct['two_electron']:.12f}")

    print("\nEnergy comparison:")
    print(f"  SCF returned:       {result.energy:.12f} Ha")
    print(f"  Fock formula:       {energy_fock:.12f} Ha")
    print(f"  Direct integral:    {energy_direct['total']:.12f} Ha")

    print(
        "  Fock vs direct:     "
        f"{abs(energy_fock - energy_direct['total']):.3e} Ha"
    )

    print(
        "  Returned vs Fock:   "
        f"{abs(result.energy - energy_fock):.3e} Ha"
    )

    # ---------------------------------------------------------
    # 7. Final orbital-Fock consistency
    # ---------------------------------------------------------

    residual_alpha = (
        F_alpha @ C_alpha
        - C_alpha * result.orbital_energies_alpha[np.newaxis, :]
    )

    residual_beta = (
        F_beta @ C_beta
        - C_beta * result.orbital_energies_beta[np.newaxis, :]
    )

    # Only occupied columns are relevant for the SCF solution.
    residual_alpha_occ = np.linalg.norm(
        residual_alpha[:, :system.n_alpha]
    )

    residual_beta_occ = np.linalg.norm(
        residual_beta[:, :system.n_beta]
    )

    print("\nOccupied orbital residuals:")
    print(f"  Alpha: {residual_alpha_occ:.3e}")
    print(f"  Beta:  {residual_beta_occ:.3e}")

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(
        "Alpha trace error:",
        f"{alpha_density['trace_error']:.3e}",
    )
    print(
        "Beta trace error: ",
        f"{beta_density['trace_error']:.3e}",
    )

    print(
        "Alpha idempotency:",
        f"{alpha_density['idempotency_error']:.3e}",
    )
    print(
        "Beta idempotency: ",
        f"{beta_density['idempotency_error']:.3e}",
    )

    print(
        "Energy disagreement:",
        f"{abs(energy_fock - energy_direct['total']):.3e} Ha",
    )

    print("=" * 75)

    return {
        "energy_fock": energy_fock,
        "energy_direct": energy_direct,
        "coulomb_ordering_difference": _matrix_error(J_rs, J_sr),
        "alpha_density": alpha_density,
        "beta_density": beta_density,
        "alpha_orbitals": alpha_orbitals,
        "beta_orbitals": beta_orbitals,
        "occupied_residual_alpha": residual_alpha_occ,
        "occupied_residual_beta": residual_beta_occ,
    }


def main():
    Z = 4.0
    B = 0.0

    N_ALPHA = 2
    N_BETA = 1

    M_VALUES = np.array([0], dtype=int)

    MAX_ORBITALS = 16
    ORBITAL_COUNTS = np.array([16], dtype=int)

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



    diagnostics = run_uhf_consistency_test(
        system=system,
        basis=basis,
        g=g,
        result=result,
    )

if __name__ == "__main__":
    main()
