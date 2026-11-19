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

    Attributes
    ----------
    m : int
        Magnetic quantum number.

    energies : ndarray, shape (n_orbitals,)
        One-electron orbital energies.

    coefficients : ndarray, shape (n_fem, n_orbitals)
        Orbital coefficients in the reduced FEM basis.

    overlap : csc_matrix
        FEM overlap matrix S_{mu,nu}.

    hamiltonian : csc_matrix
        One-electron Hamiltonian matrix h_{mu,nu} = T + V
        in the FEM basis.

    kinetic : csc_matrix
        Kinetic energy matrix T.

    potential : csc_matrix
        Potential energy matrix V.

    free_indices : ndarray
        Indices of retained nodes in the original FEM grid.

    residual_norms : ndarray
        Norms of generalized eigenvalue residuals.
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
        """
        Check C^dagger S C = I.
        """

        metric = self.coefficients.conj().T @ (
            self.overlap @ self.coefficients
        )

        identity = np.eye(self.n_orbitals)

        error = np.linalg.norm(metric - identity)

        return error < tolerance

    def orthonormality_error(self):
        """
        Return ||C^dagger S C - I||.
        """

        metric = self.coefficients.conj().T @ (
            self.overlap @ self.coefficients
        )

        identity = np.eye(self.n_orbitals)

        return np.linalg.norm(metric - identity)

    def eigenvalue_residuals(self):
        """
        Return residual vectors:

            r_p = h C_p - epsilon_p S C_p
        """

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
    Solve the one-electron generalized eigenvalue problem
    in a fixed magnetic quantum number sector.

        h C = S C epsilon

    The FEM matrices are assembled using operators.fem.

    Parameters
    ----------
    rho_nodes : array
        Physical radial mesh nodes.

    z_nodes : array
        Physical z mesh nodes.

    Z : float
        Nuclear charge.

    B : float
        Magnetic field in atomic units.

    m : int
        Magnetic quantum number.

    n_orbitals : int
        Number of lowest-energy orbitals to calculate.

    quadrature_order : int
        Gauss-Legendre quadrature order.

    sigma : float or None
        Optional shift for shift-invert eigensolver.
        If None, the lowest algebraic eigenvalues are calculated.

    tol : float
        Eigenvalue solver tolerance.

    maxiter : int or None
        Maximum number of eigensolver iterations.

    ncv : int or None
        Number of Lanczos vectors used by ARPACK.

    Returns
    -------
    OrbitalSet
        Calculated spatial orbitals and FEM matrices.
    """

    if not isinstance(m, (int, np.integer)):
        raise TypeError("m must be an integer.")

    if not isinstance(n_orbitals, (int, np.integer)):
        raise TypeError("n_orbitals must be an integer.")

    if n_orbitals <= 0:
        raise ValueError("n_orbitals must be positive.")

    if Z <= 0:
        raise ValueError("Nuclear charge must be positive.")

    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    # --------------------------------------------------
    # 1. Assemble FEM matrices
    # --------------------------------------------------

    T, S, V, free_indices = assemble_fem_matrices(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        Z=Z,
        B=B,
        m=m,
        spin=0.0,
        quadrature_order=quadrature_order,
    )

    # --------------------------------------------------
    # 2. Construct one-electron Hamiltonian
    # --------------------------------------------------

    h = (T + V).tocsc()

    n_fem = h.shape[0]

    if n_orbitals > n_fem:
        raise ValueError(
            f"Requested {n_orbitals} orbitals, "
            f"but the FEM basis contains only {n_fem} functions."
        )

    # --------------------------------------------------
    # 3. Solve generalized eigenvalue problem
    #
    #       h C = S C epsilon
    #
    # Use shift-invert for sparse matrices.
    # --------------------------------------------------

    if n_orbitals >= n_fem - 1:

        energies, coefficients = eigh(
            h.toarray(),
            S.toarray(),
            subset_by_index=(0, n_orbitals - 1),
            driver="gvx",
        )

    else:

        # Estimate a shift below the lowest hydrogen-like
        # orbital energy. The user can override it.
        if sigma is None:
            sigma = -0.6 * Z**2 - 0.5 * B * m

        # Lanczos subspace dimension.
        if ncv is None:
            ncv = min(
                n_fem,
                max(4 * n_orbitals + 1, 40),
            )

        # Maximum number of iterations.
        if maxiter is None:
            maxiter = max(2000, 20 * n_fem)

        energies, coefficients = eigsh(
            h,
            k=n_orbitals,
            M=S,
            sigma=sigma,
            which="LM",
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
        )


    # --------------------------------------------------
    # 4. Sort eigenvalues and eigenvectors
    # --------------------------------------------------

    order = np.argsort(energies)

    energies = np.asarray(energies[order])
    coefficients = np.asarray(coefficients[:, order])

    # --------------------------------------------------
    # 5. Calculate generalized eigenvalue residuals
    # --------------------------------------------------

    HC = h @ coefficients
    SC = S @ coefficients

    residuals = HC - SC * energies[np.newaxis, :]

    residual_norms = np.linalg.norm(residuals, axis=0)

    # --------------------------------------------------
    # 6. Construct orbital set
    # --------------------------------------------------

    orbitals = OrbitalSet(
        m=int(m),
        energies=energies,
        coefficients=coefficients,
        overlap=S,
        hamiltonian=h,
        kinetic=T,
        potential=V,
        free_indices=free_indices,
        residual_norms=residual_norms,
    )

    return orbitals
