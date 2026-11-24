
from dataclasses import dataclass

import numpy as np

from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

from operators.fem import assemble_fem_matrices


@dataclass
class OrbitalSet:
    """
    Set of spatial orbitals for one magnetic quantum number m.
    """

    m: int
    energies: np.ndarray
    coefficients: np.ndarray

    overlap: csc_matrix
    hamiltonian: csc_matrix

    kinetic: csc_matrix
    potential: csc_matrix

    free_indices: np.ndarray
    residual_norms: np.ndarray

    @property
    def n_orbitals(self):
        return len(self.energies)

    @property
    def n_fem(self):
        return self.coefficients.shape[0]

    def check_orthonormality(self, tolerance=1e-9):
        metric = self.coefficients.conj().T @ (
            self.overlap @ self.coefficients
        )

        identity = np.eye(self.n_orbitals)

        error = np.linalg.norm(metric - identity)

        return error < tolerance

    def orthonormality_error(self):
        metric = self.coefficients.conj().T @ (
            self.overlap @ self.coefficients
        )

        identity = np.eye(self.n_orbitals)

        return np.linalg.norm(metric - identity)

    def eigenvalue_residuals(self):
        HC = self.hamiltonian @ self.coefficients
        SC = self.overlap @ self.coefficients

        return HC - SC * self.energies[np.newaxis, :]


def solve_orbitals(
    rho_nodes,
    z_nodes,
    Z,
    B,
    m,
    n_orbitals,
    quadrature_order=8,
    sigma=None,
    tol=1e-10,
    maxiter=None,
    ncv=None,
):
    """
    Solve one-electron generalized eigenvalue problems
    for multiple magnetic quantum number sectors.

    Parameters
    ----------
    m : np.ndarray
        Magnetic quantum numbers.

    n_orbitals : np.ndarray
        Number of orbitals to calculate in each m sector.

    Other parameters are unchanged.

    Returns
    -------
    list[OrbitalSet]
        One OrbitalSet for each magnetic quantum number.
    """

    m = np.asarray(m)
    n_orbitals = np.asarray(n_orbitals)

    if m.ndim != 1:
        raise ValueError("m must be a one-dimensional array.")

    if n_orbitals.ndim != 1:
        raise ValueError("n_orbitals must be a one-dimensional array.")

    if len(m) != len(n_orbitals):
        raise ValueError(
            "m and n_orbitals must have the same length."
        )

    if len(m) == 0:
        raise ValueError("At least one m sector must be provided.")

    if not np.issubdtype(m.dtype, np.integer):
        raise TypeError("All elements of m must be integers.")

    if not np.issubdtype(n_orbitals.dtype, np.integer):
        raise TypeError("All elements of n_orbitals must be integers.")

    if np.any(n_orbitals <= 0):
        raise ValueError("All values of n_orbitals must be positive.")

    if Z <= 0:
        raise ValueError("Nuclear charge must be positive.")

    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    if len(np.unique(m)) != len(m):
        raise ValueError("Each magnetic quantum number must be unique.")

    orbital_sets = []

    for m_sector, n_sector in zip(m, n_orbitals):

        m_sector = int(m_sector)
        n_sector = int(n_sector)

        # 1. Assemble FEM matrices

        T, S, V, free_indices = assemble_fem_matrices(
            rho_nodes=rho_nodes,
            z_nodes=z_nodes,
            Z=Z,
            B=B,
            m=m_sector,
            spin=0.0,
            quadrature_order=quadrature_order,
        )

        # 2. Construct one-electron Hamiltonian

        h = (T + V).tocsc()

        n_fem = h.shape[0]

        if n_sector > n_fem:
            raise ValueError(
                f"Requested {n_sector} orbitals for m={m_sector}, "
                f"but the FEM basis contains only {n_fem} functions."
            )

        # 3. Solve generalized eigenvalue problem

        if n_sector >= n_fem - 1:

            energies, coefficients = eigh(
                h.toarray(),
                S.toarray(),
                subset_by_index=(0, n_sector - 1),
                driver="gvx",
            )

        else:

            sigma_sector = sigma

            if sigma_sector is None:
                sigma_sector = -0.6 * Z**2 - 0.5 * B * m_sector

            ncv_sector = ncv

            if ncv_sector is None:
                ncv_sector = min(
                    n_fem,
                    max(4 * n_sector + 1, 40),
                )

            maxiter_sector = maxiter

            if maxiter_sector is None:
                maxiter_sector = max(2000, 20 * n_fem)

            energies, coefficients = eigsh(
                h,
                k=n_sector,
                M=S,
                sigma=sigma_sector,
                which="LM",
                tol=tol,
                maxiter=maxiter_sector,
                ncv=ncv_sector,
            )

        # 4. Sort eigenvalues and eigenvectors

        order = np.argsort(energies)

        energies = np.asarray(energies[order])
        coefficients = np.asarray(coefficients[:, order])

        # 5. Calculate generalized eigenvalue residuals

        HC = h @ coefficients
        SC = S @ coefficients

        residuals = HC - SC * energies[np.newaxis, :]

        residual_norms = np.linalg.norm(residuals, axis=0)

        # 6. Construct orbital set

        orbital_set = OrbitalSet(
            m=m_sector,
            energies=energies,
            coefficients=coefficients,
            overlap=S,
            hamiltonian=h,
            kinetic=T,
            potential=V,
            free_indices=free_indices,
            residual_norms=residual_norms,
        )

        orbital_sets.append(orbital_set)

    return orbital_sets
