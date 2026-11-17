
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh

from grid.mesh import make_atomic_mesh
from operators.fem import assemble_fem_matrices


# =========================================================
# CONFIGURATION
# =========================================================

Z = 1
B = 0.0
M = 0
SPIN = 0.0

A_RHO = 1.0
A_Z = 1.0
B_POISSON = 0.5

QUADRATURE_ORDER = 8
N_STATES = 6

# Domain sizes
T_MAX_VALUES = [0.980, 0.990, 0.995]

# Experiment A: fixed number of intervals
FIXED_GRID_SIZES = [120, 160, 200]

# Experiment B: approximately fixed resolution near nucleus
REFERENCE_T_MAX = 0.980
REFERENCE_N = 200

# Exact hydrogen energies in the m = 0 sector
EXACT_ENERGIES = np.array([
    -0.5,
    -0.125,
    -0.125,
    -1.0 / 18.0,
    -1.0 / 18.0,
    -1.0 / 18.0,
])


# =========================================================
# MESH INFORMATION
# =========================================================

def physical_boundary(t_max, A, b):

    return A * (t_max + b) * t_max / (1.0 - t_max**2)


def near_nucleus_step(t_max, n, A, b):

    # x(t) ~ A*b*t near t = 0
    return A * b * t_max / n


# =========================================================
# CALCULATE EIGENVALUES
# =========================================================

def calculate_energies(n, t_max):

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n,
        n_z=n,
        A_rho=A_RHO,
        A_z=A_Z,
        b=B_POISSON,
        t_rho_max=t_max,
        t_z_max=t_max,
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
        sigma=-0.45 * Z**2,
        which="LM",
        tol=1e-10,
        maxiter=200000,
        ncv=40,
    )

    order = np.argsort(eigenvalues)

    return eigenvalues[order]


# =========================================================
# RUN AND CACHE CALCULATIONS
# =========================================================

calculation_cache = {}


def run_calculation(n, t_max):

    key = (n, round(t_max, 8))

    if key in calculation_cache:
        print("Using cached calculation.")
        return calculation_cache[key]

    print("\n" + "-" * 85)
    print(f"Calculating: N = {n}, t_max = {t_max:.5f}")

    energies = calculate_energies(n, t_max)

    errors = np.abs(energies - EXACT_ENERGIES)

    calculation_cache[key] = (energies, errors)

    x_rho = physical_boundary(
        t_max, A_RHO, B_POISSON
    )

    x_z = physical_boundary(
        t_max, A_Z, B_POISSON
    )

    h_rho = near_nucleus_step(
        t_max, n, A_RHO, B_POISSON
    )

    h_z = near_nucleus_step(
        t_max, n, A_Z, B_POISSON
    )

    print(f"Physical rho boundary: {x_rho:.8f}")
    print(f"Physical z boundary:   +/- {x_z:.8f}")
    print(f"Near-core rho step:    {h_rho:.8e}")
    print(f"Near-core z step:      {h_z:.8e}")

    print("\nState     Energy (Eh)           Error")

    for i in range(N_STATES):

        print(
            f"{i + 1:<10}"
            f"{energies[i]:<22.12f}"
            f"{errors[i]:.6e}"
        )

    return energies, errors


# =========================================================
# SPECTRAL ANALYSIS
# =========================================================

def calculate_splittings(energies):

    splitting_n2 = abs(energies[1] - energies[2])

    splitting_n3 = (
        np.max(energies[3:6]) -
        np.min(energies[3:6])
    )

    return splitting_n2, splitting_n3


def print_spectral_summary(energies):

    splitting_n2, splitting_n3 = calculate_splittings(
        energies
    )

    print(f"n = 2 splitting: {splitting_n2:.8e} Eh")
    print(f"n = 3 splitting: {splitting_n3:.8e} Eh")


# =========================================================
# EXPERIMENT A: FIXED GRID SIZE
# =========================================================

def run_fixed_grid_experiment():

    print("\n\n")
    print("=" * 85)
    print("EXPERIMENT A: FIXED GRID SIZE")
    print("=" * 85)

    results = {}

    for n in FIXED_GRID_SIZES:

        for t_max in T_MAX_VALUES:

            energies, errors = run_calculation(n, t_max)

            results[(n, t_max)] = energies

            print("\nSpectral splitting:")
            print_spectral_summary(energies)

    return results


# =========================================================
# EXPERIMENT B: FIXED NEAR-CORE RESOLUTION
# =========================================================

def run_fixed_resolution_experiment():

    print("\n\n")
    print("=" * 85)
    print("EXPERIMENT B: FIXED NEAR-CORE RESOLUTION")
    print("=" * 85)

    results = {}

    for t_max in T_MAX_VALUES:

        n = int(round(
            REFERENCE_N * t_max / REFERENCE_T_MAX
        ))

        energies, errors = run_calculation(n, t_max)

        results[t_max] = {
            "n": n,
            "energies": energies,
            "errors": errors,
        }

        print("\nSpectral splitting:")
        print_spectral_summary(energies)

    return results


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results_csv(fixed_results, matched_results):

    filename = "domain_convergence.csv"

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Experiment",
            "t_max",
            "N",
            "rho_max",
            "z_max",
            "state",
            "energy",
            "exact_energy",
            "absolute_error",
            "n2_splitting",
            "n3_splitting",
        ])

        for (n, t_max), energies in fixed_results.items():

            splitting_n2, splitting_n3 = (
                calculate_splittings(energies)
            )

            errors = np.abs(energies - EXACT_ENERGIES)

            for i in range(N_STATES):

                writer.writerow([
                    "fixed_grid",
                    t_max,
                    n,
                    physical_boundary(
                        t_max, A_RHO, B_POISSON
                    ),
                    physical_boundary(
                        t_max, A_Z, B_POISSON
                    ),
                    i + 1,
                    energies[i],
                    EXACT_ENERGIES[i],
                    errors[i],
                    splitting_n2,
                    splitting_n3,
                ])

        for t_max, data in matched_results.items():

            n = data["n"]
            energies = data["energies"]
            errors = data["errors"]

            splitting_n2, splitting_n3 = (
                calculate_splittings(energies)
            )

            for i in range(N_STATES):

                writer.writerow([
                    "fixed_near_core_resolution",
                    t_max,
                    n,
                    physical_boundary(
                        t_max, A_RHO, B_POISSON
                    ),
                    physical_boundary(
                        t_max, A_Z, B_POISSON
                    ),
                    i + 1,
                    energies[i],
                    EXACT_ENERGIES[i],
                    errors[i],
                    splitting_n2,
                    splitting_n3,
                ])

    print(f"\nResults saved to {filename}")


# =========================================================
# PLOTS
# =========================================================

def plot_domain_convergence(matched_results):

    t_values = sorted(matched_results.keys())

    x_values = [
        physical_boundary(t, A_RHO, B_POISSON)
        for t in t_values
    ]

    errors = np.array([
        matched_results[t]["errors"]
        for t in t_values
    ])

    # Energy errors
    plt.figure(figsize=(9, 6))

    for state in range(N_STATES):

        plt.plot(
            x_values,
            errors[:, state],
            marker="o",
            label=f"State {state + 1}",
        )

    plt.xlabel("Physical domain boundary")
    plt.ylabel("Absolute energy error (Eh)")
    plt.title("Domain convergence at fixed near-core resolution")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "domain_convergence_errors.png",
        dpi=200
    )

    # Spectral splitting
    splittings_n2 = []
    splittings_n3 = []

    for t in t_values:

        e = matched_results[t]["energies"]

        s2, s3 = calculate_splittings(e)

        splittings_n2.append(s2)
        splittings_n3.append(s3)

    plt.figure(figsize=(9, 6))

    plt.plot(
        x_values,
        splittings_n2,
        marker="o",
        label="n = 2",
    )

    plt.plot(
        x_values,
        splittings_n3,
        marker="s",
        label="n = 3",
    )

    plt.xlabel("Physical domain boundary")
    plt.ylabel("Energy splitting (Eh)")
    plt.title("Hydrogen degeneracy restoration")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "domain_convergence_splitting.png",
        dpi=200
    )

    plt.show()


def plot_fixed_grid_results(fixed_results):

    plt.figure(figsize=(9, 6))

    for n in FIXED_GRID_SIZES:

        errors = []

        for t_max in T_MAX_VALUES:

            energies = fixed_results[(n, t_max)]

            errors.append(
                np.max(
                    np.abs(energies - EXACT_ENERGIES)
                )
            )

        x_values = [
            physical_boundary(t, A_RHO, B_POISSON)
            for t in T_MAX_VALUES
        ]

        plt.plot(
            x_values,
            errors,
            marker="o",
            label=f"N = {n}",
        )

    plt.xlabel("Physical domain boundary")
    plt.ylabel("Maximum energy error (Eh)")
    plt.title("Domain dependence at fixed grid sizes")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "fixed_grid_domain_convergence.png",
        dpi=200
    )

    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 85)
    print("HYDROGEN DOMAIN AND GRID CONVERGENCE")
    print("=" * 85)

    print(f"Z = {Z}")
    print(f"B = {B}")
    print(f"m = {M}")
    print(f"Quadrature order = {QUADRATURE_ORDER}")
    print(f"Atomic Poisson b = {B_POISSON}")

    print("\nDomain parameters:", T_MAX_VALUES)

    print("\nPhysical domain boundaries:")

    for t in T_MAX_VALUES:

        print(
            f"t_max = {t:.3f}: "
            f"rho_max = "
            f"{physical_boundary(t, A_RHO, B_POISSON):.8f}, "
            f"z_max = +/- "
            f"{physical_boundary(t, A_Z, B_POISSON):.8f}"
        )

    fixed_results = run_fixed_grid_experiment()

    matched_results = run_fixed_resolution_experiment()

    save_results_csv(
        fixed_results,
        matched_results,
    )

    plot_domain_convergence(matched_results)

    plot_fixed_grid_results(fixed_results)

    print("\n" + "=" * 85)
    print("CONVERGENCE EXPERIMENT FINISHED")
    print("=" * 85)


if __name__ == "__main__":
    main()
