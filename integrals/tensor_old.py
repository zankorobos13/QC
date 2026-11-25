
import numpy as np

from integrals.poisson import integrate_poisson_orbitals


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

    Integral convention:

        g[p,q,r,s] = <pq|1/r12|rs>

        = integral phi_p*(1) phi_q*(2)
          1/r12
          phi_r(1) phi_s(2) d1 d2

    The magnetic selection rule is:

        m[p] - m[r] + m[q] - m[s] = 0

    The implementation exploits the permutation symmetries
    valid for real radial orbital functions.

    Parameters
    ----------
    basis : OrbitalBasis
        Combined orbital basis. Must provide m_values and n_orbitals.

    psi_arr : ndarray
        Radial orbital values on the FEM grid.
        Shape: (n_orbitals, n_rho, n_z).

    rho_nodes, z_nodes : ndarray
        Physical FEM grid coordinates.

    quadrature_order : int
        FEM quadrature order used by the Poisson integrator.

    symmetry_tolerance : float
        Tolerance for checking that radial orbitals are real.

    Returns
    -------
    g : ndarray
        Full four-index Coulomb tensor.
        Shape: (n_orbitals, n_orbitals, n_orbitals, n_orbitals).
    """

    m_values = np.asarray(basis.m_values)
    psi_arr = np.asarray(psi_arr)
    rho_nodes = np.asarray(rho_nodes)
    z_nodes = np.asarray(z_nodes)

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

    if psi_arr.ndim != 3:
        raise ValueError(
            "psi_arr must have shape (n_orbitals, n_rho, n_z)."
        )

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

    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    # The current FEM orbital solver produces real radial functions.
    max_imaginary = np.max(np.abs(np.imag(psi_arr)))

    if max_imaginary > symmetry_tolerance:
        raise ValueError(
            "The radial orbitals are complex. "
            "The real-orbital permutation symmetries "
            "used by this implementation are not applicable."
        )

    psi_arr = np.asarray(np.real(psi_arr), dtype=np.float64)

    # ---------------------------------------------------------
    # Allocate tensor
    # ---------------------------------------------------------

    g = np.zeros(
        (n_orbitals,) * 4,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Generate unordered orbital pairs (p, r).
    #
    # Each pair is characterized by:
    #
    #     |m[p] - m[r]|
    #
    # Only pairs with equal absolute magnetic transfer
    # can interact.
    # ---------------------------------------------------------

    pair_groups = {}

    for p in range(n_orbitals):
        for r in range(p, n_orbitals):
            k = abs(m_values[p] - m_values[r])

            pair_groups.setdefault(k, []).append((p, r))

    # ---------------------------------------------------------
    # Calculate unique pair-pair integrals
    # ---------------------------------------------------------

    for k, pairs in pair_groups.items():

        for ia, (p, r) in enumerate(pairs):

            delta_pr = m_values[p] - m_values[r]

            for q, s in pairs[ia:]:

                delta_qs = m_values[q] - m_values[s]

                # Choose the orientation of the second pair
                # so that the magnetic selection rule is satisfied.
                if delta_pr + delta_qs == 0:
                    q0, s0 = q, s
                else:
                    q0, s0 = s, q

                # -------------------------------------------------
                # Evaluate one independent integral
                # -------------------------------------------------

                value = integrate_poisson_orbitals(
                    orbital_p=psi_arr[p],
                    orbital_q=psi_arr[q0],
                    orbital_r=psi_arr[r],
                    orbital_s=psi_arr[s0],
                    rho_nodes=rho_nodes,
                    z_nodes=z_nodes,
                    k=k,
                    quadrature_order=quadrature_order,
                )

                value = complex(value)

                if abs(value.imag) > symmetry_tolerance:
                    raise ValueError(
                        f"Unexpected imaginary part in integral "
                        f"g[{p},{q0},{r},{s0}]: {value.imag}"
                    )

                value = value.real

                # -------------------------------------------------
                # Apply permutation symmetries
                # -------------------------------------------------

                base = (p, q0, r, s0)

                if k == 0:
                    # All eight permutations are valid when
                    # both transition densities have zero
                    # magnetic transfer.
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
                    # For nonzero magnetic transfer, only
                    # four permutations preserve the selection rule.
                    permutations = (
                        (p, q0, r, s0),
                        (q0, p, s0, r),
                        (r, s0, p, q0),
                        (s0, r, q0, p),
                    )

                for indices in permutations:
                    g[indices] = value

    return g
