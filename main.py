from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals

rho_mesh, z_mesh = make_atomic_mesh(
    n_rho=80,
    n_z=80,
    A_rho=1.0,
    A_z=1.0,
    b=0.5,
    t_rho_max=0.98,
    t_z_max=0.98,
)

orbitals = solve_orbitals(
    rho_nodes=rho_mesh.physical,
    z_nodes=z_mesh.physical,
    Z=1,
    B=0.0,
    m=0,
    n_orbitals=6,
)

print("Magnetic quantum number:", orbitals.m)
print("Number of orbitals:", orbitals.n_orbitals)

print("\nOrbital energies:")
print(orbitals.energies)

print("\nCoefficient matrix shape:")
print(orbitals.coefficients.shape)

print("\nOrthonormality error:")
print(orbitals.orthonormality_error())

print("\nEigenvalue residuals:")
print(orbitals.residual_norms)

# print("\nHamiltonian:")
# print(orbitals.hamiltonian)