
from dataclasses import dataclass

import numpy as np

from scf.density import build_density_matrix
from scf.fock import build_fock_matrices
from scf.energy import calculate_electronic_energy


@dataclass
class UHFResult:
    """Result of a UHF-SCF calculation."""

    energy: float
    converged: bool
    iterations: int

    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray

    coefficients_alpha: np.ndarray
    coefficients_beta: np.ndarray

    density_alpha: np.ndarray
    density_beta: np.ndarray

    fock_alpha: np.ndarray
    fock_beta: np.ndarray

    energy_history: np.ndarray
    density_error: float


def _diagonalize_by_sector(matrix, basis):
    """
    Diagonalize a Hermitian matrix independently in each m sector.

    Eigenvectors from different sectors are never mixed.
    The resulting orbitals are globally sorted by energy.
    """

    n = basis.n_orbitals
    m_values = np.asarray(basis.m_values)

    if m_values.shape != (n,):
        raise ValueError("basis.m_values has an invalid shape.")

    if not np.all(np.isfinite(m_values)):
        raise ValueError("basis.m_values contains non-finite values.")

    if not np.all(m_values == np.round(m_values)):
        raise ValueError("Magnetic quantum numbers must be integers.")

    energies = []
    coefficients = []

    for m in np.unique(m_values):
        indices = np.flatnonzero(m_values == m)

        block = matrix[np.ix_(indices, indices)]

        block_energies, block_coefficients = np.linalg.eigh(block)

        for local_index, energy in enumerate(block_energies):
            vector = np.zeros(n, dtype=np.complex128)
            vector[indices] = block_coefficients[:, local_index]

            energies.append(energy)
            coefficients.append(vector)

    energies = np.asarray(energies, dtype=float)
    coefficients = np.column_stack(coefficients)

    # Global Aufbau ordering across all m sectors.
    order = np.argsort(energies)

    return energies[order], coefficients[:, order]


def _build_initial_orbitals(h_alpha, h_beta, basis):
    """Construct the initial orbitals from one-electron Hamiltonians."""

    eps_alpha, C_alpha = _diagonalize_by_sector(h_alpha, basis)
    eps_beta, C_beta = _diagonalize_by_sector(h_beta, basis)

    return eps_alpha, C_alpha, eps_beta, C_beta


def _density_error(D_new, D_old):
    """Calculate the RMS Frobenius norm of the density change."""

    n = D_new.shape[0]

    return float(
        np.linalg.norm(D_new - D_old) / np.sqrt(n)
    )


def run_uhf(
    system,
    basis,
    g,
    max_iterations=100,
    energy_tolerance=1e-10,
    density_tolerance=1e-8,
    damping=0.0,
):
    """
    Perform an unrestricted Hartree-Fock SCF calculation.

    Parameters
    ----------
    system : ElectronicSystem
        Electronic system configuration.

    basis : OrbitalBasis
        Orthonormal spatial orbital basis.

    g : ndarray, shape (n, n, n, n)
        Two-electron Coulomb integrals.

    max_iterations : int
        Maximum number of SCF iterations.

    energy_tolerance : float
        Convergence threshold for total energy.

    density_tolerance : float
        Convergence threshold for density matrices.

    damping : float
        Density damping parameter in [0, 1).

        D_new = (1 - damping) * D_candidate + damping * D_old

    Returns
    -------
    UHFResult
        SCF energy, orbitals, densities, Fock matrices and convergence data.
    """

    if isinstance(max_iterations, bool) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise TypeError("max_iterations must be an integer.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0:
        raise ValueError("energy_tolerance must be positive and finite.")

    if not np.isfinite(density_tolerance) or density_tolerance <= 0:
        raise ValueError("density_tolerance must be positive and finite.")

    if not np.isfinite(damping) or not 0.0 <= damping < 1.0:
        raise ValueError("damping must satisfy 0 <= damping < 1.")

    system.validate_basis(basis)

    n = basis.n_orbitals

    g = np.asarray(g, dtype=np.complex128)

    if g.shape != (n, n, n, n):
        raise ValueError(
            f"Expected Coulomb tensor shape {(n, n, n, n)}, "
            f"got {g.shape}."
        )

    if not np.all(np.isfinite(g)):
        raise ValueError("Coulomb tensor contains non-finite values.")

    # Spin-dependent one-electron Hamiltonians.
    h_alpha, h_beta = system.get_one_electron_matrices(basis)

    # Initial guess: occupied eigenvectors of the one-electron Hamiltonians.
    (
        eps_alpha,
        C_alpha,
        eps_beta,
        C_beta,
    ) = _build_initial_orbitals(h_alpha, h_beta, basis)

    D_alpha = build_density_matrix(C_alpha, system.n_alpha)
    D_beta = build_density_matrix(C_beta, system.n_beta)

    energy_history = []
    previous_energy = None

    converged = False
    density_error = np.inf

    F_alpha = h_alpha.copy()
    F_beta = h_beta.copy()

    for iteration in range(1, max_iterations + 1):

        # 1. Build Fock matrices from current densities.
        (
            F_alpha,
            F_beta,
            _,
            _,
            _,
        ) = build_fock_matrices(
            h_alpha,
            h_beta,
            g,
            D_alpha,
            D_beta,
        )

        # 2. Solve the alpha and beta Roothaan equations.
        eps_alpha, C_alpha = _diagonalize_by_sector(
            F_alpha, basis
        )

        eps_beta, C_beta = _diagonalize_by_sector(
            F_beta, basis
        )

        # 3. Construct candidate density matrices.
        D_alpha_candidate = build_density_matrix(
            C_alpha,
            system.n_alpha,
        )

        D_beta_candidate = build_density_matrix(
            C_beta,
            system.n_beta,
        )

        # 4. Apply density damping, if requested.
        D_alpha_new = (
            (1.0 - damping) * D_alpha_candidate
            + damping * D_alpha
        )

        D_beta_new = (
            (1.0 - damping) * D_beta_candidate
            + damping * D_beta
        )

        density_error = max(
            _density_error(D_alpha_new, D_alpha),
            _density_error(D_beta_new, D_beta),
        )

        # 5. Rebuild Fock matrices for the updated densities.
        (
            F_alpha_new,
            F_beta_new,
            _,
            _,
            _,
        ) = build_fock_matrices(
            h_alpha,
            h_beta,
            g,
            D_alpha_new,
            D_beta_new,
        )

        # 6. Calculate the energy using mutually consistent
        #    densities and Fock matrices.
        energy = calculate_electronic_energy(
            h_alpha,
            h_beta,
            F_alpha_new,
            F_beta_new,
            D_alpha_new,
            D_beta_new,
        )

        energy_history.append(energy)

        if previous_energy is not None:
            energy_error = abs(energy - previous_energy)

            if (
                energy_error < energy_tolerance
                and density_error < density_tolerance
            ):
                converged = True

        # 7. Update densities and Fock matrices.
        D_alpha = D_alpha_new
        D_beta = D_beta_new

        F_alpha = F_alpha_new
        F_beta = F_beta_new

        previous_energy = energy

        if converged:
            break

    return UHFResult(
        energy=energy,
        converged=converged,
        iterations=iteration,
        orbital_energies_alpha=eps_alpha,
        orbital_energies_beta=eps_beta,
        coefficients_alpha=C_alpha,
        coefficients_beta=C_beta,
        density_alpha=D_alpha,
        density_beta=D_beta,
        fock_alpha=F_alpha,
        fock_beta=F_beta,
        energy_history=np.asarray(energy_history),
        density_error=density_error,
    )
