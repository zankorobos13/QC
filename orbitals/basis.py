
from dataclasses import dataclass

import numpy as np

from orbitals.solver import OrbitalSet
from orbitals.evaluate import evaluate_orbital_grid


@dataclass
class OrbitalBasis:
    """
    Unified orthonormal spatial orbital basis assembled
    from multiple magnetic quantum number sectors.

    Attributes
    ----------
    orbital_sets : list[OrbitalSet]
        Original FEM orbital sets, one for each m.

    m_values : ndarray, shape (n_orbitals,)
        Magnetic quantum number of each global orbital.

    energies : ndarray, shape (n_orbitals,)
        One-electron orbital energies.

    hamiltonian : ndarray, shape (n_orbitals, n_orbitals)
        One-electron Hamiltonian in the unified orbital basis.

    sector_indices : dict[int, ndarray]
        Global orbital indices belonging to each m sector.

    orbital_indices : list[tuple[int, int]]
        Mapping from global orbital index to
        (sector index, local orbital index).
    """

    orbital_sets: list

    m_values: np.ndarray
    energies: np.ndarray
    hamiltonian: np.ndarray

    sector_indices: dict
    orbital_indices: list

    @property
    def n_orbitals(self):
        return len(self.energies)

    def get_orbitals_psi(self, rho, z):
        psi_arr = []
        m_arr = []

        for orbital_set in self.orbital_sets:
            for j in range(orbital_set.n_orbitals):
                psi = evaluate_orbital_grid(
                    orbitals=orbital_set,
                    rho_nodes=rho,
                    z_nodes=z,
                    orbital_index=j,
                )

                psi_arr.append(psi)
                m_arr.append(orbital_set.m)

        return np.asarray(psi_arr), np.asarray(m_arr)

    def check_orthonormality(self, tolerance=1e-9):
        """
        Check orthonormality of the unified orbital basis.

        Different m sectors are orthogonal analytically.
        Within each sector, FEM orthonormality is checked.
        """

        return all(
            orbital_set.check_orthonormality(tolerance)
            for orbital_set in self.orbital_sets
        )

    def orthonormality_errors(self):
        """
        Return orthonormality errors for all m sectors.
        """

        return {
            orbital_set.m: orbital_set.orthonormality_error()
            for orbital_set in self.orbital_sets
        }

    def get_orbital(self, global_index):
        """
        Return the original OrbitalSet and local orbital index
        corresponding to a global orbital index.
        """

        if not 0 <= global_index < self.n_orbitals:
            raise IndexError("Invalid global orbital index.")

        sector_index, local_index = self.orbital_indices[global_index]

        return self.orbital_sets[sector_index], local_index

    def get_sector_indices(self, m):
        """
        Return global orbital indices belonging to a given m.
        """

        if m not in self.sector_indices:
            raise KeyError(f"No orbitals available for m={m}.")

        return self.sector_indices[m]


def build_orbital_basis(orbital_sets, orthonormality_tolerance=1e-9):
    """
    Construct a unified orthonormal orbital basis and
    transform the one-electron Hamiltonian into it.

    Parameters
    ----------
    orbital_sets : list[OrbitalSet]
        Results returned by solve_orbitals.

    orthonormality_tolerance : float
        Maximum allowed FEM orthonormality error.

    Returns
    -------
    OrbitalBasis
        Unified spatial orbital basis and h_pq matrix.
    """

    if not orbital_sets:
        raise ValueError("orbital_sets must not be empty.")

    if not all(isinstance(item, OrbitalSet) for item in orbital_sets):
        raise TypeError("All elements must be OrbitalSet instances.")

    m_values = []
    energies = []
    orbital_indices = []
    sector_indices = {}

    n_total = sum(item.n_orbitals for item in orbital_sets)

    hamiltonian = np.zeros((n_total, n_total), dtype=np.complex128)

    global_offset = 0

    for sector_index, orbital_set in enumerate(orbital_sets):

        m = orbital_set.m
        n = orbital_set.n_orbitals

        if m in sector_indices:
            raise ValueError(f"Duplicate magnetic sector m={m}.")

        # --------------------------------------------------
        # 1. Check orthonormality within the FEM sector
        # --------------------------------------------------

        error = orbital_set.orthonormality_error()

        if error > orthonormality_tolerance:
            raise ValueError(
                f"Orbitals with m={m} are not orthonormal: "
                f"error={error:.3e}"
            )

        # --------------------------------------------------
        # 2. Transform one-electron Hamiltonian
        #
        # h_MO = C^dagger H_FEM C
        # --------------------------------------------------

        C = orbital_set.coefficients
        H = orbital_set.hamiltonian

        h_sector = C.conj().T @ (H @ C)

        # --------------------------------------------------
        # 3. Insert the sector into the global matrix
        # --------------------------------------------------

        global_indices = np.arange(
            global_offset,
            global_offset + n,
        )

        hamiltonian[
            global_offset:global_offset + n,
            global_offset:global_offset + n,
        ] = h_sector

        sector_indices[m] = global_indices

        # --------------------------------------------------
        # 4. Store global orbital metadata
        # --------------------------------------------------

        m_values.extend([m] * n)
        energies.extend(orbital_set.energies)

        for local_index in range(n):
            orbital_indices.append(
                (sector_index, local_index)
            )

        global_offset += n

    # ------------------------------------------------------
    # 5. Construct unified basis
    # ------------------------------------------------------

    m_values = np.asarray(m_values, dtype=int)
    energies = np.asarray(energies, dtype=float)

    # Numerical Hermiticity check.
    if not np.allclose(
        hamiltonian,
        hamiltonian.conj().T,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Transformed one-electron Hamiltonian is not Hermitian."
        )

    return OrbitalBasis(
        orbital_sets=orbital_sets,
        m_values=m_values,
        energies=energies,
        hamiltonian=hamiltonian,
        sector_indices=sector_indices,
        orbital_indices=orbital_indices,
    )
