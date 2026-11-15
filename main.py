import numpy as np
from scipy.sparse.linalg import eigsh

from grid.mesh import make_atomic_mesh
from operators.fem import assemble_fem_matrices


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

Z = 1
B = 0.0
M = 0
SPIN = 0.0

A_RHO = 1.0
A_Z = 1.0
B_POISSON = 0.5

T_MAX = 0.995
QUADRATURE_ORDER = 8

GRID_SIZES = [40, 60, 80, 100, 120, 160, 200]

N_STATES = 6

# Exact hydrogen energies in the m = 0 sector
EXACT_ENERGIES = np.array([
    -0.5,
    -0.125,
    -0.125,
    -1.0 / 18.0,
    -1.0 / 18.0,
    -1.0 / 18.0,
])


# ---------------------------------------------------------
# CALCULATE EIGENVALUES
# ---------------------------------------------------------

def calculate_energies(n):

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n,
        n_z=n,
        A_rho=A_RHO,
        A_z=A_Z,
        b=B_POISSON,
        t_rho_max=T_MAX,
        t_z_max=T_MAX,
    )

    rho = rho_mesh.physical
    z = z_mesh.physical

    T, S, V, free_indices = assemble_fem_matrices(
        rho,
        z,
        Z=Z,
        B=B,
        m=M,
        spin=SPIN,
        quadrature_order=QUADRATURE_ORDER,
    )

    H = (T + V).tocsc()
    S = S.tocsc()

    eigenvalues, eigenvectors = eigsh(
        H,
        k=N_STATES,
        M=S,
        sigma=-0.45,
        which="LM",
        tol=1e-10,
        maxiter=200000,
        ncv=40,
    )

    return np.sort(eigenvalues)


# ---------------------------------------------------------
# CONVERGENCE ANALYSIS
# ---------------------------------------------------------

def calculate_orders(errors, grid_sizes):

    orders = []

    for i in range(len(grid_sizes) - 1):

        n1 = grid_sizes[i]
        n2 = grid_sizes[i + 1]

        e1 = errors[i]
        e2 = errors[i + 1]

        if e1 > 0.0 and e2 > 0.0:

            p = np.log(e1 / e2) / np.log(n2 / n1)

        else:
            p = np.nan

        orders.append(p)

    return orders



def extrapolate_energies(all_energies, grid_sizes):

    print("\n\nENERGY EXTRAPOLATION")
    print("-" * 85)

    x = 1.0 / np.array(grid_sizes, dtype=float)**2

    for state in range(N_STATES):

        y = all_energies[:, state]

        # Linear fit: E(N) = E_inf + C / N^2
        coefficients = np.polyfit(x, y, 1)

        C, E_inf = coefficients

        exact = EXACT_ENERGIES[state]

        print(f"\nState {state + 1}")
        print(f"Extrapolated energy: {E_inf:.12f} Eh")
        print(f"Exact energy:        {exact:.12f} Eh")
        print(f"Extrapolation error: {abs(E_inf - exact):.6e} Eh")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    all_energies = []
    all_errors = []

    print("HYDROGEN GRID CONVERGENCE")
    print("-" * 85)

    print(f"Atomic number Z: {Z}")
    print(f"Magnetic field B: {B}")
    print(f"Magnetic quantum number m: {M}")
    print(f"Domain parameter t_max: {T_MAX}")
    print(f"Quadrature order: {QUADRATURE_ORDER}")
    print(f"Atomic Poisson parameter b: {B_POISSON}")

    print("\nGrid sizes:", GRID_SIZES)

    # Calculate energies for all grids
    for n in GRID_SIZES:

        print("\n" + "-" * 85)
        print(f"Calculating grid {n} x {n} ...")

        energies = calculate_energies(n)

        errors = np.abs(energies - EXACT_ENERGIES)

        all_energies.append(energies)
        all_errors.append(errors)

        print(f"\nGrid: {n} x {n}")

        print(
            f"{'State':<10}"
            f"{'Energy (Eh)':<22}"
            f"{'Exact (Eh)':<22}"
            f"{'Absolute error':<20}"
        )

        for i in range(N_STATES):

            print(
                f"{i + 1:<10}"
                f"{energies[i]:<22.12f}"
                f"{EXACT_ENERGIES[i]:<22.12f}"
                f"{errors[i]:.6e}"
            )

    all_errors = np.array(all_errors)

    # Calculate empirical convergence orders
    print("\n\nGRID CONVERGENCE ORDERS")
    print("-" * 85)

    for state in range(N_STATES):

        orders = calculate_orders(
            all_errors[:, state],
            GRID_SIZES,
        )

        print(f"\nState {state + 1}")

        print(
            f"{'Grid transition':<20}"
            f"{'Error 1':<18}"
            f"{'Error 2':<18}"
            f"{'Order p':<12}"
        )

        for i, p in enumerate(orders):

            print(
                f"{GRID_SIZES[i]} -> {GRID_SIZES[i + 1]:<10}"
                f"{all_errors[i, state]:<18.6e}"
                f"{all_errors[i + 1, state]:<18.6e}"
                f"{p:<12.4f}"
            )

    # Final summary
    print("\n" + "-" * 85)
    print("GRID CONVERGENCE TEST FINISHED.")

    extrapolate_energies(
        np.array(all_energies),
        GRID_SIZES
    )


if __name__ == "__main__":
    main()
