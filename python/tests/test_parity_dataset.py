"""Cross-language parity tests for dataset operations (rpy2)."""

import numpy as np
import pytest

from fmridataset import matrix_dataset

pytestmark = pytest.mark.parity


def test_blocklens_parity(r_fmridataset) -> None:
    """blocklens matches between R and Python."""
    from rpy2.robjects import r as R
    from rpy2.robjects import conversion, default_converter

    with conversion.localconverter(default_converter):
        mat = np.zeros((30, 5))
        py_ds = matrix_dataset(mat, TR=1.0, run_length=[10, 10, 10])

        r_mat = R.matrix(mat.ravel(order="F").tolist(), nrow=30, ncol=5)
        r_ds = r_fmridataset.matrix_dataset(r_mat, TR=1.0, run_length=R.c(10, 10, 10))
        r_bl = np.array(r_fmridataset.blocklens(r_ds), dtype=int)

        np.testing.assert_array_equal(list(py_ds.blocklens), r_bl)


def test_n_timepoints_parity(r_fmridataset) -> None:
    """n_timepoints matches."""
    from rpy2.robjects import r as R
    from rpy2.robjects import conversion, default_converter

    with conversion.localconverter(default_converter):
        mat = np.zeros((50, 5))
        py_ds = matrix_dataset(mat, TR=1.0, run_length=[20, 30])

        r_mat = R.matrix(mat.ravel(order="F").tolist(), nrow=50, ncol=5)
        r_ds = r_fmridataset.matrix_dataset(r_mat, TR=1.0, run_length=R.c(20, 30))
        r_n = int(np.array(r_fmridataset.n_timepoints(r_ds))[0])

        assert py_ds.n_timepoints == r_n


def test_matrix_dataset_parity_does_not_call_numpy2ri_activate(r_fmridataset) -> None:
    """Parity tests should not rely on deprecated numpy2ri.activate()."""
    from rpy2.robjects import r as R
    from rpy2.robjects import conversion, default_converter
    from unittest.mock import patch

    mat = np.zeros((6, 2))

    with (
        patch("rpy2.robjects.numpy2ri.activate") as activate_fn,
        patch("rpy2.robjects.numpy2ri.deactivate") as deactivate_fn,
    ):
        with conversion.localconverter(default_converter):
            r_mat = R.matrix(mat.ravel(order="F").tolist(), nrow=6, ncol=2)
            r_ds = r_fmridataset.matrix_dataset(r_mat, TR=1.0, run_length=R.c(6))
        assert r_ds is not None

    assert not activate_fn.called
    assert not deactivate_fn.called
