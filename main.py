
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

from grid.mesh import make_atomic_mesh
from orbitals import solve_orbitals, evaluate_orbital_grid


def main():
    # ---------------------------------------------------------
    # 1. Physical parameters
    # ---------------------------------------------------------

    Z = 1
    B = 0
    m = 0

    n_orbitals = 1

    # FEM grid
    n_rho = 100
    n_z = 100

    A_rho = 1.0
    A_z = 1.0
    b = 0.5

    t_rho_max = 0.98
    t_z_max = 0.98

    # ---------------------------------------------------------
    # 2. Build computational mesh
    # ---------------------------------------------------------

    rho_mesh, z_mesh = make_atomic_mesh(
        n_rho=n_rho,
        n_z=n_z,
        A_rho=A_rho,
        A_z=A_z,
        b=b,
        t_rho_max=t_rho_max,
        t_z_max=t_z_max,
    )

    rho_nodes = rho_mesh.physical
    z_nodes = z_mesh.physical

    # ---------------------------------------------------------
    # 3. Solve the one-electron problem
    # ---------------------------------------------------------

    orbitals = solve_orbitals(
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        Z=Z,
        B=B,
        m=m,
        n_orbitals=n_orbitals,
    )

    print("ORBITAL CALCULATION")
    print("-" * 50)
    print(f"Z = {Z}")
    print(f"B = {B}")
    print(f"m = {m}")
    print(f"Ground-state energy = {orbitals.energies[0]:.10f} Eh")
    print(f"Orthonormality error = {orbitals.orthonormality_error():.3e}")
    print()

    # ---------------------------------------------------------
    # 4. Recover the orbital on the FEM grid
    # ---------------------------------------------------------

    psi_grid = evaluate_orbital_grid(
        orbitals=orbitals,
        rho_nodes=rho_nodes,
        z_nodes=z_nodes,
        orbital_index=0,
    )

    # ---------------------------------------------------------
    # 5. Interpolate the FEM orbital in cylindrical coordinates
    # ---------------------------------------------------------

    interpolator = RegularGridInterpolator(
        (rho_nodes, z_nodes),
        psi_grid,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # ---------------------------------------------------------
    # 6. Build a Cartesian 3D grid
    # ---------------------------------------------------------

    extent = 8.0 / Z
    n_plot = 65

    x = np.linspace(-extent, extent, n_plot)
    y = np.linspace(-extent, extent, n_plot)
    z = np.linspace(-extent, extent, n_plot)

    X, Y, Z_grid = np.meshgrid(
        x, y, z, indexing="ij"
    )

    RHO = np.sqrt(X**2 + Y**2)

    points = np.column_stack((
        RHO.ravel(),
        Z_grid.ravel(),
    ))

    # The spatial orbital:
    # Psi(rho, phi, z) = exp(-i*m*phi) * psi(rho,z) / sqrt(2*pi)

    psi_2d = interpolator(points).reshape(X.shape)

    PHI = np.arctan2(Y, X)

    Psi = (
        np.exp(-1j * m * PHI)
        * psi_2d
        / np.sqrt(2.0 * np.pi)
    )

    density = np.abs(Psi)**2

    # ---------------------------------------------------------
    # 7. Prepare 3D visualization
    # ---------------------------------------------------------

    threshold = 0.12 * density.max()

    mask = density > threshold

    fig = plt.figure(figsize=(15, 7))

    # ---------------------------------------------------------
    # 8. 3D probability density
    # ---------------------------------------------------------

    ax1 = fig.add_subplot(121, projection="3d")

    scatter = ax1.scatter(
        X[mask],
        Y[mask],
        Z_grid[mask],
        c=density[mask],
        cmap="viridis",
        s=5,
        alpha=0.35,
    )

    ax1.set_title(
        r"3D orbital density $|\Psi(x,y,z)|^2$"
    )

    ax1.set_xlabel("x (Bohr)")
    ax1.set_ylabel("y (Bohr)")
    ax1.set_zlabel("z (Bohr)")

    ax1.set_xlim(-extent, extent)
    ax1.set_ylim(-extent, extent)
    ax1.set_zlim(-extent, extent)

    ax1.set_box_aspect((1, 1, 1))

    fig.colorbar(
        scatter,
        ax=ax1,
        shrink=0.65,
        label="Probability density",
    )

    # ---------------------------------------------------------
    # 9. Cross-section in the xz plane (y = 0)
    # ---------------------------------------------------------

    ax2 = fig.add_subplot(122)

    x_section = np.linspace(-extent, extent, 200)
    z_section = np.linspace(-extent, extent, 200)

    XS, ZS = np.meshgrid(x_section, z_section)

    RHO_section = np.abs(XS)

    section_points = np.column_stack((
        RHO_section.ravel(),
        ZS.ravel(),
    ))

    psi_section = interpolator(
        section_points
    ).reshape(XS.shape)

    density_section = np.abs(psi_section)**2 / (2.0 * np.pi)

    contour = ax2.contourf(
        XS,
        ZS,
        density_section,
        levels=40,
        cmap="viridis",
    )

    ax2.set_title(r"Cross-section of $|\Psi(x,0,z)|^2$")
    ax2.set_xlabel("x (Bohr)")
    ax2.set_ylabel("z (Bohr)")
    ax2.set_aspect("equal")

    fig.colorbar(
        contour,
        ax=ax2,
        label="Probability density",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
