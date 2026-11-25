"""
Discrete weak-form residual test for the k=0 Poisson FEM operator.

The analytic hydrogenic 1s potential is sampled at FEM nodes and inserted
into the assembled discrete Poisson system. The RHS is assembled by the
same orbital interpolation and quadrature path as the production solver.

Run from the project root:
    python tests/test_poisson_weak_residual.py
"""

from __future__ import annotations

import numpy as np

from grid.mesh import make_atomic_mesh
from integrals.poisson import (
    _get_poisson_factorization,
    assemble_poisson_rhs_from_orbitals,
    _transition_charge,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

Z = 4.0
K = 0
QUADRATURE_ORDER = 8
GRID_SIZES = (40, 60, 80, 120, 160)

A_RHO = 1.0
A_Z = 1.0
MAPPING_B = 0.5
T_RHO_MAX = 0.98
T_Z_MAX = 0.98


# ---------------------------------------------------------------------------
# Analytic functions
# ---------------------------------------------------------------------------

def hydrogenic_1s(rho_nodes, z_nodes, nuclear_charge):
    """Radial part psi of phi=(2*pi)^(-1/2)*psi for a hydrogenic 1s orbital."""
    radius = np.sqrt(
        np.asarray(rho_nodes)[:, None] ** 2
        + np.asarray(z_nodes)[None, :] ** 2
    )
    return np.sqrt(2.0 * nuclear_charge**3) * np.exp(-nuclear_charge * radius)


def analytic_1s_potential(rho_nodes, z_nodes, nuclear_charge):
    r"""V(r) = [1 - (1 + Z*r)*exp(-2*Z*r)]/r, with V(0)=Z."""
    radius = np.sqrt(
        np.asarray(rho_nodes)[:, None] ** 2
        + np.asarray(z_nodes)[None, :] ** 2
    )
    x = nuclear_charge * radius

    potential = np.empty_like(radius, dtype=np.float64)
    small = x < 1.0e-4
    regular = ~small

    xs = x[small]
    potential[small] = nuclear_charge * (
        1.0
        - 2.0 * xs**2 / 3.0
        + 2.0 * xs**3 / 3.0
        - 2.0 * xs**4 / 5.0
    )

    xr = x[regular]
    rr = radius[regular]
    numerator = -np.expm1(-2.0 * xr) - xr * np.exp(-2.0 * xr)
    potential[regular] = numerator / rr

    return potential


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def vector_norms(residual, rhs):
    residual = np.asarray(residual)
    rhs = np.asarray(rhs)

    norm_r = float(np.linalg.norm(residual))
    norm_b = float(np.linalg.norm(rhs))
    max_r = float(np.max(np.abs(residual))) if residual.size else 0.0

    relative = norm_r / norm_b if norm_b > 0.0 else np.nan
    return norm_r, max_r, relative


def run_grid(n_rho, n_z):
    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n_rho,
        n_z=n_z,
        A_rho=A_RHO,
        A_z=A_Z,
        b=MAPPING_B,
        t_rho_max=T_RHO_MAX,
        t_z_max=T_Z_MAX,
    )

    rho_nodes = np.asarray(rho_mesh.physical, dtype=np.float64)
    z_nodes = np.asarray(z_mesh.physical, dtype=np.float64)

    # Keep the exact, continuously normalized hydrogenic orbital here.
    # Do not renormalize it using the finite-grid quadrature: the analytic
    # potential below corresponds to the continuum-normalized density.
    psi = hydrogenic_1s(rho_nodes, z_nodes, Z)
    potential_exact = analytic_1s_potential(rho_nodes, z_nodes, Z)

    # Retrieve the exact production matrix, factorization and free indices.
    lu, K_free, free_indices, K_full = _get_poisson_factorization(
        rho_nodes,
        z_nodes,
        K,
        QUADRATURE_ORDER,
    )

    # Assemble the RHS using precisely the production orbital-pair path.
    rhs = assemble_poisson_rhs_from_orbitals(
        orbital_q=psi,
        orbital_s=psi,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        free_indices=free_indices,
        quadrature_order=QUADRATURE_ORDER,
    )

    n_total = len(rho_nodes) * len(z_nodes)
    all_indices = np.arange(n_total)
    is_free = np.zeros(n_total, dtype=bool)
    is_free[free_indices] = True
    boundary_indices = all_indices[~is_free]

    exact_vector = potential_exact.ravel()
    exact_free = exact_vector[free_indices]
    exact_boundary = exact_vector[boundary_indices]

    # Full free-row residual includes the contribution from prescribed
    # Dirichlet boundary values.
    residual = (
        K_free @ exact_free
        + K_full[free_indices, :][:, boundary_indices] @ exact_boundary
        - rhs
    )

    residual_l2, residual_max, relative_residual = vector_norms(
        residual, rhs
    )

        # Reconstruct the same boundary conditions as the production solver.
    boundary_values = np.zeros(
        (len(rho_nodes), len(z_nodes)),
        dtype=np.float64,
    )

    discrete_charge = _transition_charge(
        orbital_q=psi,
        orbital_s=psi,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        quadrature_order=QUADRATURE_ORDER,
    )

    rho_grid = rho_nodes[:, None]
    z_grid = z_nodes[None, :]
    radius = np.sqrt(rho_grid**2 + z_grid**2)

    boundary_mask = np.zeros(
        (len(rho_nodes), len(z_nodes)),
        dtype=bool,
    )
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    boundary_values[boundary_mask] = (
        discrete_charge.real / radius[boundary_mask]
    )

    boundary_vector = boundary_values.ravel()[boundary_indices]

    # Production Dirichlet correction.
    rhs_corrected = rhs - (
        K_full[free_indices, :][:, boundary_indices]
        @ boundary_vector
    )

    # Solve the same discrete problem as solve_poisson_orbital_pair().
    numerical_free = lu.solve(rhs_corrected)

    potential_error = numerical_free - exact_free

    max_potential_error = float(np.max(np.abs(potential_error)))
    rms_potential_error = float(
        np.linalg.norm(potential_error) / np.sqrt(potential_error.size)
    )

    potential_error = numerical_free - exact_free

    max_potential_error = float(np.max(np.abs(potential_error)))
    rms_potential_error = float(
        np.linalg.norm(potential_error) / np.sqrt(potential_error.size)
    )


    return {
        "grid": f"{n_rho}x{n_z}",
        "n_free": len(free_indices),
        "charge": complex(discrete_charge),
        "residual_l2": residual_l2,
        "residual_max": residual_max,
        "relative_residual": relative_residual,
        "potential_max_error": max_potential_error,
        "potential_rms_error": rms_potential_error,
    }


def main():
    print("=" * 116)
    print("HYDROGENIC 1s: DISCRETE POISSON WEAK-FORM RESIDUAL")
    print("=" * 116)
    print(f"Z = {Z:g}")
    print(f"k = {K}")
    print(f"Quadrature order = {QUADRATURE_ORDER}")
    print("Analytic potential is sampled at nodes; exact Dirichlet values are imposed.")
    print("RHS uses the production orbital interpolation and quadrature.")
    print()

    results = [run_grid(n, n) for n in GRID_SIZES]

    print(
        f"{'Grid':>10} {'N_free':>10} {'Charge':>14} "
        f"{'||r||_2':>14} {'max|r_i|':>14} {'||r||/||b||':>16}"
    )
    print("-" * 116)
    for result in results:
        print(
            f"{result['grid']:>10} "
            f"{result['n_free']:10d} "
            f"{result['charge'].real:14.10f} "
            f"{result['residual_l2']:14.6e} "
            f"{result['residual_max']:14.6e} "
            f"{result['relative_residual']:16.6e}"
        )

    print()
    print("NUMERICAL SOLUTION ERROR AGAINST ANALYTIC NODAL POTENTIAL")
    print("-" * 116)
    print(
        f"{'Grid':>10} {'max|V_h-V_exact|':>22} "
        f"{'RMS(V_h-V_exact)':>22}"
    )
    print("-" * 116)
    for result in results:
        print(
            f"{result['grid']:>10} "
            f"{result['potential_max_error']:22.6e} "
            f"{result['potential_rms_error']:22.6e}"
        )

    print()
    print("Notes:")
    print("  The residual is K_ff*V_exact,f + K_fb*V_exact,b - b_f.")
    print("  A decreasing residual under mesh refinement is expected for a consistent Q1 FEM.")
    print("  The charge is not forced to one: the analytic continuum orbital is used unchanged.")
    print("=" * 116)


if __name__ == "__main__":
    main()
