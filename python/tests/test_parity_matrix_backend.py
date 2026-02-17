"""Cross-language parity tests for matrix backend / matrix_dataset (rpy2)."""

import numpy as np
import pytest

from fmridataset import matrix_dataset

pytestmark = pytest.mark.parity


def test_matrix_dataset_roundtrip(r_fmridataset) -> None:
    """matrix_dataset(mat) -> get_data() matches in R and Python."""
    from rpy2.robjects import r as R
    from rpy2.robjects import conversion, default_converter

    with conversion.localconverter(default_converter):
        rng = np.random.default_rng(99)
        mat = rng.standard_normal((20, 10))

        # Python
        py_ds = matrix_dataset(mat, TR=2.0, run_length=20)
        py_data = py_ds.get_data()

        # R
        r_mat = R.matrix(mat.ravel(order="F").tolist(), nrow=20, ncol=10)
        r_ds = r_fmridataset.matrix_dataset(r_mat, TR=2.0, run_length=20)
        r_data = np.array(r_fmridataset.get_data(r_ds)).reshape(20, 10, order="F")

        np.testing.assert_array_almost_equal(py_data, r_data)
