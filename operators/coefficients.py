from dataclasses import dataclass

import numpy as np

from grid.mesh import Mesh1D


@dataclass
class GridCoefficients:
    """
    Coefficients on the tensor-product (t, u) grid.

    Array convention:
        axis 0 -> radial coordinate t
        axis 1 -> longitudinal coordinate u
    """

    rho: np.ndarray
    z: np.ndarray

    jacobian_rho: np.ndarray
    jacobian_z: np.ndarray

    measure: np.ndarray

    kinetic_rho: np.ndarray
    kinetic_z: np.ndarray
    centrifugal: np.ndarray

    quadrature_weights: np.ndarray


def trapezoidal_weights(mesh: Mesh1D) -> np.ndarray:
    """
    Trapezoidal quadrature weights on a uniform mathematical grid.
    """
    n = mesh.size

    if n < 2:
        raise ValueError("At least two grid points are required")

    weights = np.full(n, mesh.step, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    return weights


def build_grid_coefficients(
    rho_mesh: Mesh1D,
    z_mesh: Mesh1D,
    m: int,
) -> GridCoefficients:
    """
    Build coefficients of the transformed kinetic operator
    and the integration measure.

    The Coulomb potential is deliberately not included here.
    """

    if not isinstance(m, (int, np.integer)):
        raise TypeError("m must be an integer")

    # Coordinates: shape (N_rho, N_z)
    rho = rho_mesh.physical[:, None]
    z = z_mesh.physical[None, :]

    # Jacobians: broadcast to the full 2D grid
    jr = rho_mesh.jacobian[:, None]
    jz = z_mesh.jacobian[None, :]

    # Cylindrical volume element after coordinate transformation:
    # dV = rho * J_rho * J_z * dt * du
    measure = rho * jr * jz

    # Coefficients in the weak kinetic-energy expression
    kinetic_rho = rho * jz / jr
    kinetic_z = rho * jr / jz

    # Centrifugal contribution.
    # At rho = 0 it is singular for m != 0.
    centrifugal = np.zeros_like(measure)

    if m != 0:
        np.divide(
            m**2 * jr * jz,
            rho,
            out=centrifugal,
            where=(rho > 0),
        )
        centrifugal = np.where(rho > 0, centrifugal, np.inf)

    # Tensor-product trapezoidal quadrature weights
    wr = trapezoidal_weights(rho_mesh)
    wz = trapezoidal_weights(z_mesh)

    quadrature_weights = (
        wr[:, None] * wz[None, :] * measure
    )

    return GridCoefficients(
        rho=rho,
        z=z,
        jacobian_rho=jr,
        jacobian_z=jz,
        measure=measure,
        kinetic_rho=kinetic_rho,
        kinetic_z=kinetic_z,
        centrifugal=centrifugal,
        quadrature_weights=quadrature_weights,
    )


def regular_magnetic_potential(
    rho: np.ndarray,
    m: int,
    B: float,
) -> np.ndarray:
    """
    Regular magnetic part of the spatial one-electron potential:

        V_B = -B*m/2 + B^2*rho^2/8

    Spin Zeeman term B*s_z is not included.
    """

    return -0.5 * B * m + 0.125 * B**2 * rho**2
