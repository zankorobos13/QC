
import gc
import platform
import sys
import time

import numpy as np
import scipy

try:
    import psutil
except ImportError:
    psutil = None

from grid.mesh import make_atomic_mesh
from orbitals.solver import solve_orbitals
from orbitals.basis import build_orbital_basis
from integrals.tensor import build_coulomb_tensor


# =============================================================
# Benchmark configuration
# =============================================================

Z = 2.0
B = 0.0

M_VALUES = np.array([-1, 0, 1], dtype=int)
N_ORBITALS = np.array([1, 2, 1], dtype=int)

N_RHO = 80
N_Z = 80

QUADRATURE_ORDER = 8

ENERGY_TOLERANCE = 5e-3
COULOMB_TOLERANCE = 3e-2
SYMMETRY_TOLERANCE = 1e-10
NORMALIZATION_TOLERANCE = 3e-2


# =============================================================
# Timing and memory utilities
# =============================================================

def memory_usage_mb():
    if psutil is None:
        return None

    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def print_memory(label):
    memory = memory_usage_mb()

    if memory is None:
        print(f"{label:<35} RSS unavailable (install psutil)")
    else:
        print(f"{label:<35} {memory:10.2f} MB")


def run_stage(name, function):
    print(f"\n[{name}]")

    memory_before = memory_usage_mb()
    start = time.perf_counter()

    result = function()

    elapsed = time.perf_counter() - start
    memory_after = memory_usage_mb()

    print(f"Time: {elapsed:.3f} s")

    if memory_before is not None:
        delta = memory_after - memory_before

        print(f"RSS before: {memory_before:.2f} MB")
        print(f"RSS after : {memory_after:.2f} MB")
        print(f"RSS change: {delta:+.2f} MB")

    return result, elapsed


# =============================================================
# Benchmark utilities
# =============================================================

def print_check(name, value, reference, tolerance):
    error = abs(value - reference)
    passed = error <= tolerance

    print(
        f"{name:<35}"
        f"{value:>16.10f}"
        f"{reference:>16.10f}"
        f"{error:>14.3e}"
        f"  {'PASS' if passed else 'FAIL'}"
    )

    return passed


def count_independent_integrals(m_values):
    """
    Count independent integrals according to the pair grouping
    used in build_coulomb_tensor.
    """

    groups = {}

    n = len(m_values)

    for p in range(n):
        for r in range(p, n):
            k = abs(m_values[p] - m_values[r])
            groups.setdefault(k, []).append((p, r))

    counts = {}

    for k, pairs in groups.items():
        counts[k] = len(pairs) * (len(pairs) + 1) // 2

    return counts


def calculate_orbital_norm(psi, rho, z):
    """
    Approximate integral:
        integral |psi(rho,z)|^2 rho drho dz.
    """

    density = np.abs(psi) ** 2 * rho[:, None]

    integral_z = np.trapezoid(
        density,
        z,
        axis=1,
    )

    return np.trapezoid(
        integral_z,
        rho,
    )


# =============================================================
# Main benchmark
# =============================================================

def main():

    total_start = time.perf_counter()
    total_memory_start = memory_usage_mb()

    stage_times = {}
    results = []

    print("=" * 90)
    print("HELIUM TWO-ELECTRON COULOMB TENSOR BENCHMARK")
    print("=" * 90)

    # ---------------------------------------------------------
    # 1. Environment
    # ---------------------------------------------------------

    print("\nENVIRONMENT")
    print("-" * 90)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"NumPy        : {np.__version__}")
    print(f"SciPy        : {scipy.__version__}")
    print(f"OS           : {platform.platform()}")
    print(f"CPU          : {platform.processor()}")
    print(f"Z            : {Z}")
    print(f"B            : {B}")
    print(f"m sectors    : {M_VALUES}")
    print(f"Orbitals     : {N_ORBITALS}")
    print(f"Total orbitals: {np.sum(N_ORBITALS)}")
    print(f"Mesh         : {N_RHO + 1} x {N_Z + 1}")
    print(f"Quadrature   : {QUADRATURE_ORDER}")

    # ---------------------------------------------------------
    # 2. Mesh
    # ---------------------------------------------------------

    (rho_mesh, z_mesh), elapsed = run_stage(
        "1. Constructing FEM mesh",
        lambda: make_atomic_mesh(
            n_rho=N_RHO,
            n_z=N_Z,
            A_rho=1.0,
            A_z=1.0,
            b=0.5,
            t_rho_max=0.98,
            t_z_max=0.98,
        ),
    )

    stage_times["Mesh"] = elapsed

    rho = rho_mesh.physical
    z = z_mesh.physical

    # ---------------------------------------------------------
    # 3. One-electron orbital calculations
    # ---------------------------------------------------------

    orbital_sets, elapsed = run_stage(
        "2. Solving one-electron problems",
        lambda: solve_orbitals(
            rho_nodes=rho,
            z_nodes=z,
            Z=Z,
            B=B,
            m=M_VALUES,
            n_orbitals=N_ORBITALS,
            quadrature_order=QUADRATURE_ORDER,
        ),
    )

    stage_times["Orbital solver"] = elapsed

    # ---------------------------------------------------------
    # 4. Build unified orbital basis
    # ---------------------------------------------------------

    basis, elapsed = run_stage(
        "3. Building unified orbital basis",
        lambda: build_orbital_basis(orbital_sets),
    )

    stage_times["Orbital basis"] = elapsed

    n_orbitals = basis.n_orbitals

    print("\nORBITAL ENERGIES")
    print("-" * 90)

    for i, energy in enumerate(basis.energies):
        print(
            f"Orbital {i:2d} | "
            f"m = {basis.m_values[i]:2d} | "
            f"E = {energy: .10f} Ha"
        )

    # ---------------------------------------------------------
    # 5. Orthonormality
    # ---------------------------------------------------------

    print("\nORTHONORMALITY")
    print("-" * 90)

    orth_errors = basis.orthonormality_errors()

    for m, error in orth_errors.items():
        print(f"m = {m:2d} | Error = {error:.3e}")

    orthonormal = basis.check_orthonormality()

    print(
        f"Unified basis orthonormality: "
        f"{'PASS' if orthonormal else 'FAIL'}"
    )

    results.append(orthonormal)

    # ---------------------------------------------------------
    # 6. Evaluate orbitals on the FEM grid
    # ---------------------------------------------------------

    (psi_arr, m_arr), elapsed = run_stage(
        "4. Evaluating orbital functions",
        lambda: basis.get_orbitals_psi(rho, z),
    )

    stage_times["Orbital evaluation"] = elapsed

    print(f"\nOrbital array shape: {psi_arr.shape}")
    print(f"Orbital m values   : {m_arr}")

    if psi_arr.shape != (n_orbitals, len(rho), len(z)):
        raise RuntimeError("Unexpected orbital array shape.")

    if not np.array_equal(m_arr, basis.m_values):
        raise RuntimeError("Orbital m values do not match the basis.")

    # ---------------------------------------------------------
    # 7. Orbital normalization
    # ---------------------------------------------------------

    print("\nORBITAL NORMALIZATION")
    print("-" * 90)

    normalization_errors = []

    for i in range(n_orbitals):
        norm = calculate_orbital_norm(
            psi_arr[i],
            rho,
            z,
        )

        error = abs(norm - 1.0)
        normalization_errors.append(error)

        print(
            f"Orbital {i:2d} | "
            f"m = {m_arr[i]:2d} | "
            f"Norm = {norm:.10f} | "
            f"Error = {error:.3e}"
        )

    normalization_passed = (
        max(normalization_errors) <= NORMALIZATION_TOLERANCE
    )

    print(
        f"Normalization: "
        f"{'PASS' if normalization_passed else 'FAIL'}"
    )

    results.append(normalization_passed)

    # ---------------------------------------------------------
    # 8. Estimate tensor size and integral count
    # ---------------------------------------------------------

    print("\nTENSOR ESTIMATE")
    print("-" * 90)

    tensor_elements = n_orbitals ** 4
    tensor_bytes = tensor_elements * np.dtype(np.float64).itemsize

    print(f"Tensor dimensions : {n_orbitals}^4")
    print(f"Tensor elements   : {tensor_elements}")
    print(f"Tensor dtype      : float64")
    print(f"Tensor memory     : {tensor_bytes / 1024**2:.6f} MB")

    integral_counts = count_independent_integrals(
        basis.m_values
    )

    for k, count in sorted(integral_counts.items()):
        print(f"k = {k:2d} | Independent integrals = {count}")

    expected_integrals = sum(integral_counts.values())

    print(f"Total independent integrals: {expected_integrals}")

    # ---------------------------------------------------------
    # 9. Build full Coulomb tensor
    # ---------------------------------------------------------

    g, elapsed = run_stage(
        "5. Building full Coulomb tensor",
        lambda: build_coulomb_tensor(
            basis=basis,
            psi_arr=psi_arr,
            rho_nodes=rho,
            z_nodes=z,
            quadrature_order=QUADRATURE_ORDER,
            symmetry_tolerance=SYMMETRY_TOLERANCE,
        ),
    )

    stage_times["Coulomb tensor"] = elapsed

    print(f"\nTensor shape: {g.shape}")
    print(f"Tensor dtype: {g.dtype}")
    print(f"Tensor nbytes: {g.nbytes / 1024**2:.6f} MB")

    if g.shape != (n_orbitals,) * 4:
        raise RuntimeError("Unexpected Coulomb tensor shape.")

    # ---------------------------------------------------------
    # 10. Physical reference values
    # ---------------------------------------------------------

    print("\nPHYSICAL BENCHMARKS")
    print("-" * 90)

    # The lowest one-electron orbital of He+ has energy -Z^2/2.
    E_exact = -Z**2 / 2.0

    ground_orbital = int(np.argmin(basis.energies))
    E_calculated = basis.energies[ground_orbital]

    # For a hydrogenic 1s orbital:
    # <1s 1s | 1/r12 | 1s 1s> = 5Z/8.
    g_exact = 5.0 * Z / 8.0

    g_calculated = g[
        ground_orbital,
        ground_orbital,
        ground_orbital,
        ground_orbital,
    ]

    print(
        f"{'Quantity':<35}"
        f"{'Calculated':>16}"
        f"{'Exact':>16}"
        f"{'Error':>14}"
        f"  Status"
    )

    print("-" * 90)

    results.append(
        print_check(
            "He+ ground-state energy",
            E_calculated,
            E_exact,
            ENERGY_TOLERANCE,
        )
    )

    results.append(
        print_check(
            "Ground orbital g_pppp",
            g_calculated,
            g_exact,
            COULOMB_TOLERANCE,
        )
    )

    # ---------------------------------------------------------
    # 11. Magnetic selection rule
    # ---------------------------------------------------------

    print("\nMAGNETIC SELECTION RULE")
    print("-" * 90)

    m = basis.m_values

    allowed = (
        m[:, None, None, None]
        + m[None, :, None, None]
        ==
        m[None, None, :, None]
        + m[None, None, None, :]
    )

    forbidden = ~allowed

    n_allowed = int(np.count_nonzero(allowed))
    n_forbidden = int(np.count_nonzero(forbidden))

    max_forbidden = np.max(np.abs(g[forbidden]))

    print(f"Allowed elements   : {n_allowed}")
    print(f"Forbidden elements : {n_forbidden}")
    print(f"Max forbidden |g|  : {max_forbidden:.3e}")

    selection_passed = max_forbidden < SYMMETRY_TOLERANCE

    print(
        f"Selection rule: "
        f"{'PASS' if selection_passed else 'FAIL'}"
    )

    results.append(selection_passed)

    # ---------------------------------------------------------
    # 12. Tensor symmetries
    # ---------------------------------------------------------

    print("\nCOULOMB TENSOR SYMMETRIES")
    print("-" * 90)

    exchange_error = np.max(
        np.abs(g - g.transpose(1, 0, 3, 2))
    )

    hermitian_error = np.max(
        np.abs(g - g.transpose(2, 3, 0, 1).conj())
    )

    print(f"Electron exchange error : {exchange_error:.3e}")
    print(f"Hermiticity error       : {hermitian_error:.3e}")

    exchange_passed = exchange_error < SYMMETRY_TOLERANCE
    hermitian_passed = hermitian_error < SYMMETRY_TOLERANCE

    results.append(exchange_passed)
    results.append(hermitian_passed)

    # ---------------------------------------------------------
    # 13. Timing report
    # ---------------------------------------------------------

    total_elapsed = time.perf_counter() - total_start

    print("\nTIMING REPORT")
    print("-" * 90)

    for name, elapsed in stage_times.items():
        print(f"{name:<35} {elapsed:12.3f} s")

    print("-" * 90)
    print(f"{'Total runtime':<35} {total_elapsed:12.3f} s")

    # ---------------------------------------------------------
    # 14. Memory report
    # ---------------------------------------------------------

    print("\nMEMORY REPORT")
    print("-" * 90)

    print_memory("Current process RSS")

    if total_memory_start is not None:
        total_memory_end = memory_usage_mb()

        print(
            f"{'RSS change since benchmark start':<35}"
            f"{total_memory_end - total_memory_start:+10.2f} MB"
        )

    if psutil is None:
        print("For RSS measurements install: pip install psutil")

    # ---------------------------------------------------------
    # 15. Final report
    # ---------------------------------------------------------

    print("\n" + "=" * 90)

    passed = sum(results)
    total = len(results)

    print(f"Checks passed: {passed}/{total}")

    if all(results):
        print("ALL BENCHMARK CHECKS PASSED")
    else:
        print("SOME BENCHMARK CHECKS FAILED")

    print("=" * 90)

    gc.collect()


if __name__ == "__main__":
    main()
