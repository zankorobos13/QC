import numpy as np
import matplotlib.pyplot as plt

from grid.mesh import make_atomic_mesh


def main():

    # Demonstration parameters, not yet optimized
    n_rho = 80
    n_z = 80

    A_rho = 1.0
    A_z = 1.0
    b = 0.5

    t_rho_max = 0.95
    t_z_max = 0.95

    rho, z = make_atomic_mesh(
        n_rho=n_rho,
        n_z=n_z,
        A_rho=A_rho,
        A_z=A_z,
        b=b,
        t_rho_max=t_rho_max,
        t_z_max=t_z_max
    )

    print("Number of rho nodes:", rho.size)
    print("Number of z nodes:", z.size)

    print("\nPhysical rho range:")
    print(rho.physical[0], rho.physical[-1])

    print("\nPhysical z range:")
    print(z.physical[0], z.physical[-1])

    print("\nFirst rho intervals:")
    print(np.diff(rho.physical[:6]))

    print("\nCentral z intervals:")
    center = z.size // 2
    print(np.diff(z.physical[center-2:center+3]))

    # Plot physical grid spacing
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rho.physical, np.zeros_like(rho.physical), "o")
    axes[0].set_title("Radial mesh")
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_yticks([])

    axes[1].plot(z.physical, np.zeros_like(z.physical), "o")
    axes[1].set_title("Axial mesh")
    axes[1].set_xlabel(r"$z$")
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
