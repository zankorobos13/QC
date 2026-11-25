
import numpy as np

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals


# ============================================================
# Coulomb problem benchmark
# ============================================================

B = 0.0
M = np.array([0], dtype=int)
N_ORBITALS = np.array([1], dtype=int)

QUADRATURE_ORDER = 8

# Number of FEM intervals in each direction.
GRID_SIZES = [40, 60, 80, 120, 160]

# Atomic mesh parameters.
A_RHO = 1.0
A_Z = 1.0
B_MAP = 0.5

T_RHO_MAX = 0.98
T_Z_MAX = 0.98

NUCLEAR_CHARGES = [1, 2, 4]


def exact_energy(Z):
    """Exact ground-state energy of a hydrogen-like atom."""
    return -0.5 * Z**2


def run_single_calculation(Z, n_rho, n_z):
    """Run one-electron FEM calculation."""

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

    energy = float(orbital_set.energies[0])
    reference = exact_energy(Z)

    absolute_error = abs(energy - reference)
    relative_error = absolute_error / abs(reference)

    orthonormality_error = orbital_set.orthonormality_error()

    residuals = orbital_set.eigenvalue_residuals()
    residual_norm = float(np.linalg.norm(residuals[:, 0]))

    return {
        "energy": energy,
        "reference": reference,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "orthonormality_error": orthonormality_error,
        "residual_norm": residual_norm,
        "n_fem": orbital_set.n_fem,
    }


def main():

    print("=" * 100)
    print("HYDROGEN-LIKE ATOM: COULOMB FEM BENCHMARK")
    print("=" * 100)

    print(f"B = {B}")
    print(f"m = {M[0]}")
    print(f"Quadrature order = {QUADRATURE_ORDER}")
    print(f"Grid sizes = {GRID_SIZES}")

    for Z in NUCLEAR_CHARGES:

        print("\n" + "=" * 100)
        print(f"NUCLEAR CHARGE Z = {Z}")
        print(f"Exact ground-state energy = {exact_energy(Z):.12f} Ha")
        print("=" * 100)

        header = (
            f"{'Grid':>12}"
            f"{'N_FEM':>12}"
            f"{'E_FEM':>20}"
            f"{'Error':>16}"
            f"{'Rel. error':>16}"
            f"{'Ortho. error':>16}"
            f"{'Residual':>16}"
        )

        print(header)
        print("-" * len(header))

        previous_error = None

        for n in GRID_SIZES:

            result = run_single_calculation(
                Z=Z,
                n_rho=n,
                n_z=n,
            )

            error = result["absolute_error"]

            print(
                f"{n:>5}x{n:<5}"
                f"{result['n_fem']:>12}"
                f"{result['energy']:>20.12f}"
                f"{error:>16.6e}"
                f"{result['relative_error']:>16.6e}"
                f"{result['orthonormality_error']:>16.6e}"
                f"{result['residual_norm']:>16.6e}"
            )

            if previous_error is not None:
                if error > previous_error:
                    print(
                        "WARNING: absolute error increased "
                        "after mesh refinement."
                    )

            previous_error = error

    print("\n" + "=" * 100)
    print("BENCHMARK FINISHED")
    print("=" * 100)


if __name__ == "__main__":
    main()
