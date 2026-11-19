import numpy as np

from operators.fem import shape_functions


def _build_full_coefficients(orbitals, n_rho, n_z):
    """
    Restore coefficients on the complete grid.

    Coefficients on excluded Dirichlet nodes are zero.
    """

    coefficients = np.asarray(orbitals.coefficients)

    expected_shape = (len(orbitals.free_indices), orbitals.n_orbitals)

    if coefficients.shape != expected_shape:
        raise ValueError(
            f"Unexpected coefficient shape: {coefficients.shape}; "
            f"expected {expected_shape}."
        )

    full = np.zeros(
        (n_rho * n_z, orbitals.n_orbitals),
        dtype=coefficients.dtype,
    )

    full[orbitals.free_indices, :] = coefficients

    return full.reshape(n_rho, n_z, orbitals.n_orbitals)


def _locate_element(nodes, x):
    """
    Find the element containing x and its local coordinate.

    Returns
    -------
    element_index : int
    local_coordinate : float in [-1, 1]
    """

    if x < nodes[0] or x > nodes[-1]:
        raise ValueError(
            f"Coordinate {x} is outside [{nodes[0]}, {nodes[-1]}]."
        )

    i = np.searchsorted(nodes, x, side="right") - 1

    # The final node belongs to the last element.
    i = min(i, len(nodes) - 2)

    x_left = nodes[i]
    x_right = nodes[i + 1]

    local = 2.0 * (x - x_left) / (x_right - x_left) - 1.0

    return i, local


def evaluate_orbital(
    orbitals,
    rho_nodes,
    z_nodes,
    rho,
    z,
    orbital_index=0,
):
    """
    Evaluate a 2D FEM orbital psi_m(rho, z) at one point.

    Parameters
    ----------
    orbitals : OrbitalSet
        Result returned by solve_orbitals.

    rho_nodes, z_nodes : array
        Physical mesh coordinates.

    rho, z : float
        Point where the orbital is evaluated.

    orbital_index : int
        Orbital number, starting from zero.

    Returns
    -------
    complex or float
        Value of the 2D orbital.
    """

    rho_nodes = np.asarray(rho_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)

    if not 0 <= orbital_index < orbitals.n_orbitals:
        raise IndexError("Invalid orbital_index.")

    if rho < 0:
        raise ValueError("rho cannot be negative.")

    n_rho = len(rho_nodes)
    n_z = len(z_nodes)

    full_coefficients = _build_full_coefficients(
        orbitals, n_rho, n_z
    )

    i, xi = _locate_element(rho_nodes, rho)
    j, eta = _locate_element(z_nodes, z)

    N, _, _ = shape_functions(xi, eta)

    # Local global-grid indices.
    nodes = (
        (i, j),
        (i + 1, j),
        (i, j + 1),
        (i + 1, j + 1),
    )

    local_coefficients = np.array([
        full_coefficients[a, b, orbital_index]
        for a, b in nodes
    ])

    return np.dot(N, local_coefficients)


def evaluate_orbital_grid(
    orbitals,
    rho_nodes,
    z_nodes,
    orbital_index=0,
):
    """
    Return orbital values at all physical mesh nodes.

    Shape of result: (n_rho, n_z).

    For Q1 elements, nodal values are exactly the
    corresponding FEM coefficients.
    """

    if not 0 <= orbital_index < orbitals.n_orbitals:
        raise IndexError("Invalid orbital_index.")

    full_coefficients = _build_full_coefficients(
        orbitals,
        len(rho_nodes),
        len(z_nodes),
    )

    return full_coefficients[:, :, orbital_index].copy()


def _evaluate_from_full(
    full_coefficients,
    rho_nodes,
    z_nodes,
    rho,
    z,
    orbital_index,
):
    i, xi = _locate_element(rho_nodes, rho)
    j, eta = _locate_element(z_nodes, z)

    N, _, _ = shape_functions(xi, eta)

    local_coefficients = np.array([
        full_coefficients[i, j, orbital_index],
        full_coefficients[i + 1, j, orbital_index],
        full_coefficients[i, j + 1, orbital_index],
        full_coefficients[i + 1, j + 1, orbital_index],
    ])

    return np.dot(N, local_coefficients)


def evaluate_wavefunction(
    orbitals,
    rho_nodes,
    z_nodes,
    rho,
    phi,
    z,
    orbital_index=0,
):
    """
    Evaluate the full 3D spatial wavefunction.
    """

    rho, phi, z = np.broadcast_arrays(rho, phi, z)

    if not 0 <= orbital_index < orbitals.n_orbitals:
        raise IndexError("Invalid orbital_index.")

    full_coefficients = _build_full_coefficients(
        orbitals,
        len(rho_nodes),
        len(z_nodes),
    )

    psi_2d = np.empty(rho.shape, dtype=complex)

    for index in np.ndindex(rho.shape):
        psi_2d[index] = _evaluate_from_full(
            full_coefficients,
            rho_nodes,
            z_nodes,
            float(rho[index]),
            float(z[index]),
            orbital_index,
        )

    angular_factor = (
        np.exp(-1j * orbitals.m * phi)
        / np.sqrt(2.0 * np.pi)
    )

    return angular_factor * psi_2d