import numpy as np

from integrals.coulomb import build_full_coefficients


def test_build_full_coefficients():
    coefficients = np.array([10, 20, 30])
    free_indices = np.array([0, 2, 4])

    result = build_full_coefficients(
        coefficients=coefficients,
        free_indices=free_indices,
        n_total=5,
    )

    expected = np.array([10, 0, 20, 0, 30])

    np.testing.assert_array_equal(result, expected)