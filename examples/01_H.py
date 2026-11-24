from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals
from orbitals.basis import build_orbital_basis
import numpy as np

rho, z = make_atomic_mesh(
    n_rho=80,
    n_z=80,
    A_rho=1.0,
    A_z=1.0,
    b=0.5,
    t_rho_max=0.98,
    t_z_max=0.98,
)

orbital_sets = solve_orbitals(
    rho_nodes=rho.physical,
    z_nodes=z.physical,
    Z=1,
    B=0,
    m=np.array([-2, -1, 0, 1, 2]),
    n_orbitals=np.array([4, 6, 8, 6, 4])
)

basis = build_orbital_basis(orbital_sets)

print("Ham:")

print(basis.hamiltonian)