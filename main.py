from integrals.coulomb import *
from grid.mesh import *
from orbitals.solver import *
from integrals.poisson import *

rho_mesh, z_mesh = make_atomic_mesh(
    n_rho=80,
    n_z=80,
    A_rho=1,
    A_z=1,
    b=0.5,
    t_rho_max=0.98,
    t_z_max=0.98
)

rho, z = rho_mesh.physical, z_mesh.physical

orbitals = solve_orbitals(
    rho_nodes=rho,
    z_nodes=z,
    Z=1,
    B=0,
    m=0,
    n_orbitals=6
)


K, free_indices = assemble_poisson_matrix(
    rho_nodes=rho,
    z_nodes=z,
    k=0
)

print(K.shape)
print(len(free_indices))

diff = K - K.T

symmetry_error = (
    np.max(np.abs(diff.data))
    if diff.nnz > 0
    else 0.0
)

print(symmetry_error)
