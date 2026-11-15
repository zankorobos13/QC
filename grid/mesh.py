import numpy as np
from dataclasses import dataclass


@dataclass
class Mesh1D:
    """One-dimensional Atomic Poisson mesh."""

    mathematical: np.ndarray
    physical: np.ndarray
    jacobian: np.ndarray
    step: float

    @property
    def size(self):
        return len(self.physical)


def atomic_poisson_map(t, A, b):
    """
    Transform mathematical coordinates into physical coordinates.

    x = A * (|t| + b) * t / (1 - t^2)
    """

    t = np.asarray(t, dtype=float)

    if A <= 0:
        raise ValueError("A must be positive")

    if b <= 0:
        raise ValueError("b must be positive")

    if np.any(np.abs(t) >= 1):
        raise ValueError("All coordinates must satisfy |t| < 1")

    return A * (np.abs(t) + b) * t / (1.0 - t**2)



def atomic_poisson_jacobian(t, A, b):
    """
    Calculate dx/dt for Atomic Poisson transformation.

    J(t) = A * (b + 2|t| + b*t^2) / (1 - t^2)^2
    """

    t = np.asarray(t, dtype=float)

    numerator = b + 2.0 * np.abs(t) + b * t**2
    denominator = (1.0 - t**2)**2

    return A * numerator / denominator



def make_mesh(n_intervals, A, b, t_min, t_max):
    """
    Generate a uniform mathematical mesh and its
    nonuniform physical counterpart.

    n_intervals: number of intervals
    A, b: Atomic Poisson parameters
    t_min, t_max: mathematical coordinate boundaries
    """

    if n_intervals < 1:
        raise ValueError("n_intervals must be positive")

    if not (-1.0 < t_min < t_max < 1.0):
        raise ValueError("Require -1 < t_min < t_max < 1")

    t = np.linspace(t_min, t_max, n_intervals + 1)

    h = (t_max - t_min) / n_intervals

    x = atomic_poisson_map(t, A, b)
    jacobian = atomic_poisson_jacobian(t, A, b)

    if np.any(np.diff(x) <= 0):
        raise RuntimeError("Physical mesh is not strictly increasing")

    return Mesh1D(
        mathematical=t,
        physical=x,
        jacobian=jacobian,
        step=h
    )


def make_atomic_mesh(n_rho, n_z, A_rho, A_z, b,
                     t_rho_max, t_z_max):
    """
    Construct cylindrical (rho, z) mesh.

    rho >= 0
    -z_max <= z <= z_max
    """

    rho_mesh = make_mesh(
        n_intervals=n_rho,
        A=A_rho,
        b=b,
        t_min=0.0,
        t_max=t_rho_max
    )

    z_mesh = make_mesh(
        n_intervals=n_z,
        A=A_z,
        b=b,
        t_min=-t_z_max,
        t_max=t_z_max
    )

    return rho_mesh, z_mesh
