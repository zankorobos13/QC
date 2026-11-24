import numpy as np

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals
from orbitals.evaluate import evaluate_orbital_grid
from integrals.poisson import integrate_poisson_orbitals


# ============================================================
# Configuration: helium atom, B = 0
# ============================================================

Z = 2.0
B = 0.0
M = 0

# 80 x 80 gives a useful first convergence check.
N_RHO = 80
N_Z = 80

A_RHO = 1.0
A_Z = 1.0
B_MESH = 0.5

T_RHO_MAX = 0.98
T_Z_MAX = 0.98

ORBITAL_QUADRATURE_ORDER = 8
POISSON_QUADRATURE_ORDER = 8


def main():
    # --------------------------------------------------------
    # 1. Generate the (rho, z) mesh
    # --------------------------------------------------------

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=N_RHO,
        n_z=N_Z,
        A_rho=A_RHO,
        A_z=A_Z,
        b=B_MESH,
        t_rho_max=T_RHO_MAX,
        t_z_max=T_Z_MAX,
    )

    rho = rho_mesh.physical
    z = z_mesh.physical

    # --------------------------------------------------------
    # 2. Solve the one-electron He+ problem
    # --------------------------------------------------------

    orbitals = solve_orbitals(
        rho_nodes=rho,
        z_nodes=z,
        Z=Z,
        B=B,
        m=M,
        n_orbitals=1,
        quadrature_order=ORBITAL_QUADRATURE_ORDER,
    )

    print("Орбиталь посчитана")

    epsilon_1s = float(orbitals.energies[0])

    # --------------------------------------------------------
    # 3. Evaluate the 1s orbital on the FEM grid
    # --------------------------------------------------------

    psi_1s = evaluate_orbital_grid(
        orbitals=orbitals,
        rho_nodes=rho,
        z_nodes=z,
        orbital_index=0,
    )

    print("Волновая функция восстановлена")

    # --------------------------------------------------------
    # 4. Calculate g_1111 using the Poisson method
    #
    # g_1111 = <11 | 1/r_12 | 11>
    #
    # For m_p = m_q = m_r = m_s = 0:
    # k = m_p - m_r = 0.
    # --------------------------------------------------------

    g_1111 = integrate_poisson_orbitals(
        orbital_p=psi_1s,
        orbital_q=psi_1s,
        orbital_r=psi_1s,
        orbital_s=psi_1s,
        rho_nodes=rho,
        z_nodes=z,
        k=0,
        quadrature_order=POISSON_QUADRATURE_ORDER,
    )

    print("g_1111 получен")

    g_1111 = float(np.real_if_close(g_1111))

    # --------------------------------------------------------
    # 5. Reference value for the hydrogenic 1s orbital
    #
    # <1s 1s | 1/r_12 | 1s 1s> = 5 Z / 8.
    # This is a reference for an exact hydrogenic orbital,
    # not the exact result for the numerically represented FEM
    # orbital.
    # --------------------------------------------------------

    g_analytic = 5.0 * Z / 8.0
    relative_error = abs(g_1111 - g_analytic) / abs(g_analytic)

    # Frozen-orbital two-electron energy (not a self-consistent
    # Hartree-Fock result).
    E_one_electron = 2.0 * epsilon_1s
    E_frozen = E_one_electron + g_1111

    # --------------------------------------------------------
    # 6. Print results
    # --------------------------------------------------------

    print("=" * 60)
    print("Helium atom: g_1111 by the Poisson method")
    print("=" * 60)
    print(f"Z                              : {Z:.1f}")
    print(f"Magnetic field                 : {B:.1f} a.u.")
    print(f"Magnetic quantum number m      : {M}")
    print(f"Mesh                           : {N_RHO} x {N_Z}")
    print(f"Orbital quadrature order       : {ORBITAL_QUADRATURE_ORDER}")
    print(f"Poisson quadrature order       : {POISSON_QUADRATURE_ORDER}")
    print("-" * 60)
    print(f"He+ 1s orbital energy          : {epsilon_1s:.10f} Ha")
    print(f"g_1111 (Poisson)               : {g_1111:.10f} Ha")
    print(f"Hydrogenic reference 5Z/8      : {g_analytic:.10f} Ha")
    print(f"Relative error                 : {relative_error:.3e}")
    print("-" * 60)
    print(f"One-electron contribution 2e1s : {E_one_electron:.10f} Ha")
    print(f"Frozen-orbital energy          : {E_frozen:.10f} Ha")
    print("-" * 60)
    print(f"Orbital S-orthonormality error : "
          f"{orbitals.orthonormality_error():.3e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
