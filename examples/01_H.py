from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals

rho, z = make_atomic_mesh(
    n_rho=80,
    n_z=80,
    A_rho=1.0,
    A_z=1.0,
    b=0.5,
    t_rho_max=0.98,
    t_z_max=0.98,
)

orbitals = solve_orbitals(
    rho_nodes=rho.physical,
    z_nodes=z.physical,
    Z=1,
    B=0,
    m=0,
    n_orbitals=6
)

print("Orbital energies:")

print(orbitals.energies)

C = orbitals.coefficients
H = orbitals.hamiltonian

h = C.conj().T @ H @ C

print("Orbital Hamiltonian:")
print(h)