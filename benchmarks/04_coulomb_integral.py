
import numpy as np

from grid.mesh import make_atomic_mesh

from integrals.poisson import (
    solve_poisson_orbital_pair,
    integrate_orbital_pair_potential,
    _transition_charge,
)


# ============================================================
# Configuration
# ============================================================

Z = 4
K = 0

GRID_SIZES = [40, 60, 80, 120, 160]

QUADRATURE_ORDER = 8

A_RHO = 1.0
A_Z = 1.0
B_MAP = 0.5

T_RHO_MAX = 0.98
T_Z_MAX = 0.98


# ============================================================
# Exact hydrogen-like 1s orbital
# ============================================================

def hydrogen_1s_radial(rho_nodes, z_nodes, Z):

    rho = np.asarray(rho_nodes)[:, None]
    z = np.asarray(z_nodes)[None, :]

    radius = np.sqrt(rho**2 + z**2)

    return np.sqrt(2.0 * Z**3) * np.exp(-Z * radius)


# ============================================================
# Exact Coulomb integral
# ============================================================

def exact_coulomb_integral(Z):

    return 5.0 * Z / 8.0


# ============================================================
# Single calculation
# ============================================================

def run_calculation(n_rho, n_z):

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n_rho,
        n_z=n_z,
        A_rho=A_RHO,
        A_z=A_Z,
        b=B_MAP,
        t_rho_max=T_RHO_MAX,
        t_z_max=T_Z_MAX,
    )

    rho_nodes = rho_mesh.physical
    z_nodes = z_mesh.physical

    # Analytic normalized 1s orbital.
    psi = hydrogen_1s_radial(
        rho_nodes,
        z_nodes,
        Z,
    )

    # --------------------------------------------------------
    # 1. Check orbital normalization
    # --------------------------------------------------------

        # --------------------------------------------------------
    # 1. Check and correct orbital normalization
    # --------------------------------------------------------

    norm = _transition_charge(
        orbital_q=psi,
        orbital_s=psi,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        quadrature_order=QUADRATURE_ORDER,
    )

    if abs(norm.imag) > 1e-12:
        raise ValueError(f"Unexpected imaginary norm: {norm}")

    norm = float(norm.real)

    psi = psi / np.sqrt(norm)

    # Verify normalization.
    normalized_norm = _transition_charge(
        orbital_q=psi,
        orbital_s=psi,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        quadrature_order=QUADRATURE_ORDER,
    )

    # --------------------------------------------------------
    # 2. Solve Poisson equation
    # --------------------------------------------------------

    potential = solve_poisson_orbital_pair(
        orbital_q=psi,
        orbital_s=psi,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        k=K,
        quadrature_order=QUADRATURE_ORDER,
    )

    # --------------------------------------------------------
    # 3. Calculate Coulomb integral
    # --------------------------------------------------------

    value = integrate_orbital_pair_potential(
        orbital_p=psi,
        orbital_r=psi,
        potential=potential,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        quadrature_order=QUADRATURE_ORDER,
    )

    reference = exact_coulomb_integral(Z)

    error = value - reference

    return {
        "n_fem": (len(rho_nodes) - 2) * (len(z_nodes) - 2),
        "norm": normalized_norm,
        "value": value,
        "reference": reference,
        "error": error,
        "absolute_error": abs(error),
        "potential_min": np.min(potential),
        "potential_max": np.max(potential),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 105)
    print("HYDROGEN-LIKE ATOM: COULOMB INTEGRAL BENCHMARK")
    print("=" * 105)

    print(f"Z = {Z}")
    print(f"k = {K}")
    print(f"Quadrature order = {QUADRATURE_ORDER}")
    print(f"Exact Coulomb integral = {exact_coulomb_integral(Z):.12f} Ha")
    print()

    header = (
        f"{'Grid':>10}"
        f"{'N_FEM':>10}"
        f"{'Norm':>15}"
        f"{'g1111':>18}"
        f"{'Exact':>15}"
        f"{'Error':>15}"
    )

    print(header)
    print("-" * len(header))

    results = []

    for n in GRID_SIZES:

        result = run_calculation(n, n)
        results.append(result)

        print(
            f"{n:4d}x{n:<4d}\t"
            f"{result['n_fem']:10d}\t"
            f"{result['norm'].real:15.10f}\t"
            f"{result['value']:18.10f}\t"
            f"{result['reference']:15.10f}\t"
            f"{result['error']:15.6e}\t"
        )

    print()
    print("=" * 105)
    print("POISSON POTENTIAL DIAGNOSTICS")
    print("=" * 105)

    for n, result in zip(GRID_SIZES, results):

        print(f"\nGrid: {n}x{n}")

        print(
            "Normalization:",
            f"{result['norm'].real:.12f}"
        )

        print(
            "Potential minimum:",
            f"{result['potential_min']:.8e}"
        )

        print(
            "Potential maximum:",
            f"{result['potential_max']:.8e}"
        )

        print(
            "Absolute integral error:",
            f"{result['absolute_error']:.8e} Ha"
        )

    print()
    print("=" * 105)
    print("BENCHMARK FINISHED")
    print("=" * 105)


if __name__ == "__main__":
    main()
