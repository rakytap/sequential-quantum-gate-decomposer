"""Tiny right-side kernels must retain dense matrix multiplication semantics."""

import numpy as np
import pytest

from squander.gates import gates_Wrapper as gates


@pytest.mark.parametrize("width,target", [
    (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
    (4, 0), (4, 1), (4, 2), (4, 3),
])
@pytest.mark.parametrize("row_count", [1, 3, None])
def test_small_u3_right_kernel_matches_dense_product(width, target, row_count):
    dimension = 1 << width
    rows = dimension if row_count is None else row_count
    rng = np.random.default_rng(20261009 + 17 * width + target)
    original = np.asarray(
        rng.normal(size=(rows, dimension))
        + 1j * rng.normal(size=(rows, dimension)), dtype=np.complex128,
    )
    operation = gates.U3(width, target)
    for angles in ((0.37, -0.49, 0.91), (-1.13, 2.07, -0.63), (0.0, 0.0, 0.0)):
        parameters = np.asarray(angles, dtype=np.float64)
        expected = original @ operation.get_Matrix(parameters)
        actual = original.copy()
        operation.apply_from_right(actual, parameters=parameters)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
