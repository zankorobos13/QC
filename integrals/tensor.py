
import numpy as np

from integrals.poisson import (
    solve_poisson_orbital_pair,
    integrate_orbital_pair_potential,
)


def build_coulomb_tensor(
    basis,
    psi_arr,
    rho_nodes,
    z_nodes,
    quadrature_order=8,
    symmetry_tolerance=1e-10,
):
    """
    Build the full two-electron Coulomb integral tensor.

    Convention:

        g[p,q,r,s] = <pq|1/r12|rs>

    The magnetic selection rule is:

        m[p] - m[r] + m[q] - m[s] = 0

    Poisson potentials are calculated once per unique unordered
    orbital pair and reused for all compatible integrals.
    """

    m_values = np.asarray(basis.m_values)
    psi_arr = np.asarray(psi_arr)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    n_orbitals = basis.n_orbitals

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    if m_values.shape != (n_orbitals,):
        raise ValueError(
            "basis.m_values must contain one m value per orbital."
        )

    if not np.all(np.equal(m_values, np.rint(m_values))):
        raise ValueError("Magnetic quantum numbers must be integers.")

    m_values = m_values.astype(int)

    expected_shape = (
        n_orbitals,
        len(rho_nodes),
        len(z_nodes),
    )

    if psi_arr.shape != expected_shape:
        raise ValueError(
            f"Expected psi_arr shape {expected_shape}, "
            f"got {psi_arr.shape}."
        )

    if not np.all(np.isfinite(psi_arr)):
        raise ValueError("psi_arr contains non-finite values.")

    if not isinstance(quadrature_order, (int, np.integer)):
        raise TypeError("quadrature_order must be an integer.")

    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    if not np.isfinite(symmetry_tolerance) or symmetry_tolerance < 0:
        raise ValueError("symmetry_tolerance must be non-negative.")

    max_imaginary = np.max(np.abs(np.imag(psi_arr)))

    if max_imaginary > symmetry_tolerance:
        raise ValueError(
            "The radial orbitals are complex. "
            "Real-orbital permutation symmetries cannot be used."
        )

    psi_arr = np.asarray(np.real(psi_arr), dtype=np.float64)

    # ---------------------------------------------------------
    # Generate unordered orbital pairs grouped by |delta_m|
    # ---------------------------------------------------------

    pair_groups = {}

    for p in range(n_orbitals):
        for r in range(p, n_orbitals):

            k = abs(m_values[p] - m_values[r])

            pair_groups.setdefault(k, []).append((p, r))

    # ---------------------------------------------------------
    # Precompute one Poisson potential per unique pair.
    #
    # For real radial orbitals:
    #
    #   psi_p * psi_r = psi_r * psi_p
    #
    # The radial Poisson equation depends on k^2, so both
    # orientations of an unordered pair use the same potential.
    # ---------------------------------------------------------

    potentials = {}

    for k, pairs in pair_groups.items():

        for q, s in pairs:

            potentials[(q, s)] = solve_poisson_orbital_pair(
                orbital_q=psi_arr[q],
                orbital_s=psi_arr[s],
                rho_nodes=rho_nodes,
                z_nodes=z_nodes,
                k=k,
                quadrature_order=quadrature_order,
            )

    # ---------------------------------------------------------
    # Allocate full tensor
    # ---------------------------------------------------------

    g = np.zeros(
        (n_orbitals,) * 4,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Calculate integrals using precomputed potentials
    # ---------------------------------------------------------

    for k, pairs in pair_groups.items():

        for ia, (p, r) in enumerate(pairs):

            delta_pr = m_values[p] - m_values[r]

            for q, s in pairs[ia:]:

                delta_qs = m_values[q] - m_values[s]

                # Orient the second pair to satisfy:
                # delta_pr + delta_qs = 0.
                if delta_pr + delta_qs == 0:
                    q0, s0 = q, s
                else:
                    q0, s0 = s, q

                # The potential is invariant under pair reversal
                # for real radial orbitals.
                potential_qs = potentials[(q, s)]

                value = integrate_orbital_pair_potential(
                    orbital_p=psi_arr[p],
                    orbital_r=psi_arr[r],
                    potential=potential_qs,
                    rho_nodes=rho_nodes,
                    z_nodes=z_nodes,
                    quadrature_order=quadrature_order,
                )

                value = complex(value)

                if abs(value.imag) > symmetry_tolerance:
                    raise ValueError(
                        f"Unexpected imaginary part in "
                        f"g[{p},{q0},{r},{s0}]: {value.imag}"
                    )

                value = value.real

                # -------------------------------------------------
                # Apply permutation symmetries
                # -------------------------------------------------

                if k == 0:

                    permutations = (
                        (p, q0, r, s0),
                        (r, q0, p, s0),
                        (p, s0, r, q0),
                        (r, s0, p, q0),
                        (q0, p, s0, r),
                        (q0, r, s0, p),
                        (s0, p, q0, r),
                        (s0, r, q0, p),
                    )

                else:

                    permutations = (
                        (p, q0, r, s0),
                        (q0, p, s0, r),
                        (r, s0, p, q0),
                        (s0, r, q0, p),
                    )

                for indices in permutations:
                    g[indices] = value

    return g
