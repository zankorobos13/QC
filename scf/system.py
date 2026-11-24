from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElectronicSystem:
    """
    Electronic system configuration for UHF-SCF.

    Parameters
    ----------
    Z : int or float
        Nuclear charge.

    n_alpha : int
        Number of alpha-spin electrons.

    n_beta : int
        Number of beta-spin electrons.

    B : float
        Magnetic field along the z axis, in atomic units.
    """

    Z: float
    n_alpha: int
    n_beta: int
    B: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.Z) or self.Z <= 0:
            raise ValueError("Z must be positive and finite.")

        if isinstance(self.n_alpha, bool) or not isinstance(
            self.n_alpha, (int, np.integer)
        ):
            raise TypeError("n_alpha must be an integer.")

        if isinstance(self.n_beta, bool) or not isinstance(
            self.n_beta, (int, np.integer)
        ):
            raise TypeError("n_beta must be an integer.")

        if self.n_alpha < 0 or self.n_beta < 0:
            raise ValueError("Electron counts cannot be negative.")

        if not np.isfinite(self.B):
            raise ValueError("B must be finite.")

    @property
    def n_electrons(self):
        return self.n_alpha + self.n_beta

    @property
    def multiplicity_projection(self):
        return 0.5 * (self.n_alpha - self.n_beta)

    def validate_basis(self, basis):
        """
        Check whether the electronic configuration fits the basis.
        """

        if not hasattr(basis, "n_orbitals"):
            raise TypeError("Invalid orbital basis.")

        n_orbitals = basis.n_orbitals

        if self.n_alpha > n_orbitals:
            raise ValueError(
                "Not enough spatial orbitals for alpha electrons."
            )

        if self.n_beta > n_orbitals:
            raise ValueError(
                "Not enough spatial orbitals for beta electrons."
            )

    def get_one_electron_matrices(self, basis):
        """
        Construct spin-dependent one-electron Hamiltonians.

        The spatial Hamiltonian already contains the orbital
        magnetic contribution. The spin Zeeman term is added here:

            h_alpha = h + B/2 * I
            h_beta  = h - B/2 * I

        The spatial basis is assumed to be orthonormal.
        """

        self.validate_basis(basis)

        h = np.asarray(basis.hamiltonian, dtype=np.complex128)

        expected_shape = (basis.n_orbitals, basis.n_orbitals)

        if h.shape != expected_shape:
            raise ValueError(
                f"Expected Hamiltonian shape {expected_shape}, "
                f"got {h.shape}."
            )

        if not np.all(np.isfinite(h)):
            raise ValueError("Hamiltonian contains non-finite values.")

        if not np.allclose(h, h.conj().T, atol=1e-10):
            raise ValueError("Hamiltonian must be Hermitian.")

        identity = np.eye(basis.n_orbitals, dtype=np.complex128)

        h_alpha = h + 0.5 * self.B * identity
        h_beta = h - 0.5 * self.B * identity

        return h_alpha, h_beta
