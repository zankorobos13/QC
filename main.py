import numpy as np

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals
from orbitals.evaluate import evaluate_orbital_grid
from integrals.coulomb import (
    transition_density,
    integrate_transition_densities,
)


# ============================================================
# Configuration
# ============================================================

Z = 2.0
B = 0.0
M = 0

N_RHO = 16
N_Z = 16

A_RHO = 1.0
A_Z = 1.0
B_MESH = 0.5

T_RHO_MAX = 0.98
T_Z_MAX = 0.98

ORBITAL_QUADRATURE_ORDER = 8
COULOMB_QUADRATURE_ORDER = 2
ANGULAR_QUADRATURE_ORDER = 24


# ============================================================
# 1. Generate mesh
# ============================================================

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


# ============================================================
# 2. Solve one-electron problem (He+)
# ============================================================

orbitals = solve_orbitals(
    rho_nodes=rho,
    z_nodes=z,
    Z=Z,
    B=B,
    m=M,
    n_orbitals=1,
    quadrature_order=ORBITAL_QUADRATURE_ORDER,
)

epsilon = orbitals.energies[0]


# ============================================================
# 3. Evaluate 1s orbital on FEM nodes
# ============================================================

psi = evaluate_orbital_grid(
    orbitals=orbitals,
    rho_nodes=rho,
    z_nodes=z,
    orbital_index=0,
)


# ============================================================
# 4. Construct transition density
# ============================================================

density = transition_density(psi, psi)


# ============================================================
# 5. Calculate Coulomb integral g_1111
# ============================================================

g1111 = integrate_transition_densities(
    f_pr=density,
    f_qs=density,
    rho_nodes=rho,
    z_nodes=z,
    k=0,
    quadrature_order=COULOMB_QUADRATURE_ORDER,
    angular_quadrature_order=ANGULAR_QUADRATURE_ORDER,
)


# ============================================================
# 6. Calculate total energy
# ============================================================

E_one_electron = 2.0 * epsilon
E_electron_electron = g1111.real

E_total = E_one_electron + E_electron_electron


# ============================================================
# 7. Print results
# ============================================================

print("=" * 50)
print("Helium atom")
print("=" * 50)

print(f"Z                       : {Z}")
print(f"Magnetic field          : {B}")
print(f"Number of orbitals      : {orbitals.n_orbitals}")

print("-" * 50)

print(f"Orbital energy          : {epsilon:.10f} Ha")
print(f"One-electron energy     : {E_one_electron:.10f} Ha")
print(f"Coulomb integral        : {E_electron_electron:.10f} Ha")

print("-" * 50)

print(f"Total energy            : {E_total:.10f} Ha")
print(f"Orthonormality error    : {orbitals.orthonormality_error():.3e}")

print("=" * 50)