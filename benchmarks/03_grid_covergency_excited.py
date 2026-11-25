
import numpy as np

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals


# ============================================================
# CONFIGURATION
# ============================================================

Z = 4
B = 0.0

M = np.array([0], dtype=int)
N_ORBITALS = np.array([3], dtype=int)

QUADRATURE_ORDER = 8

GRID_SIZES = [40, 60, 80, 120, 160]

A_RHO = 1.0
A_Z = 1.0
B_MAP = 0.5

T_RHO_MAX = 0.98
T_Z_MAX = 0.98

EXACT_ENERGIES = np.array([
    -8.0,
    -2.0,
    -2.0,
])


# ============================================================
# CALCULATION
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

    orbital_sets = solve_orbitals(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        Z=Z,
        B=B,
        m=M,
        n_orbitals=N_ORBITALS,
        quadrature_order=QUADRATURE_ORDER,
    )

    orbital_set = orbital_sets[0]

    energies = orbital_set.energies[:3]

    errors = energies - EXACT_ENERGIES

    residuals = orbital_set.eigenvalue_residuals()
    residual_norms = np.linalg.norm(residuals[:, :3], axis=0)

    return {
        "energies": energies,
        "errors": errors,
        "absolute_errors": np.abs(errors),
        "degeneracy_splitting": abs(energies[2] - energies[1]),
        "orthonormality_error": orbital_set.orthonormality_error(),
        "residual_norms": residual_norms,
        "n_fem": orbital_set.n_fem,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print("HYDROGEN-LIKE ATOM: EXCITED STATES FEM BENCHMARK")
    print("=" * 110)

    print(f"Z = {Z}")
    print(f"B = {B}")
    print(f"m = 0")
    print(f"Quadrature order = {QUADRATURE_ORDER}")
    print(f"Exact energies = {EXACT_ENERGIES}")
    print()

    header = (
        f"{'Grid':>10}"
        f"{'N_FEM':>10}"
        f"{'E1':>17}"
        f"{'E2':>17}"
        f"{'E3':>17}"
        f"{'Err E1':>13}"
        f"{'Err E2':>13}"
        f"{'Err E3':>13}"
        f"{'Split':>13}"
    )

    print(header)
    print("-" * len(header))

    results = []

    for n in GRID_SIZES:

        result = run_calculation(n, n)
        results.append(result)

        e = result["energies"]
        err = result["absolute_errors"]

        print(
            f"{n:4d}x{n:<4d}"
            f"{result['n_fem']:10d}"
            f"{e[0]:17.9f}"
            f"{e[1]:17.9f}"
            f"{e[2]:17.9f}"
            f"{err[0]:13.3e}"
            f"{err[1]:13.3e}"
            f"{err[2]:13.3e}"
            f"{result['degeneracy_splitting']:13.3e}"
        )

    print()
    print("=" * 110)
    print("NUMERICAL DIAGNOSTICS")
    print("=" * 110)

    for n, result in zip(GRID_SIZES, results):

        print(f"\nGrid: {n}x{n}")

        print(
            "Orthonormality error:",
            f"{result['orthonormality_error']:.3e}"
        )

        for i, residual in enumerate(result["residual_norms"]):
            print(
                f"Residual state {i + 1}:",
                f"{residual:.3e}"
            )

        print(
            "2s/2p degeneracy splitting:",
            f"{result['degeneracy_splitting']:.6e} Ha"
        )

    print()
    print("=" * 110)
    print("BENCHMARK FINISHED")
    print("=" * 110)


if __name__ == "__main__":
    main()
