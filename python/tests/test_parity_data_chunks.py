"""Cross-language parity tests for data_chunks (rpy2)."""

import numpy as np
import pytest

from fmridataset import data_chunks, matrix_dataset

pytestmark = pytest.mark.parity


def test_single_chunk_matches_full_data(r_fmridataset) -> None:
    """A single chunk should return the same data in R and Python."""
    from rpy2.robjects import r as R
    from rpy2.robjects import conversion, default_converter

    with conversion.localconverter(default_converter):
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((20, 10))

        # Python
        py_ds = matrix_dataset(mat, TR=1.0, run_length=20)
        py_chunk = data_chunks(py_ds, nchunks=1).collect()[0]

        # R
        r_mat = R.matrix(mat.ravel(order="F").tolist(), nrow=20, ncol=10)
        r_ds = r_fmridataset.matrix_dataset(r_mat, TR=1.0, run_length=20)
        r_chunks = r_fmridataset.data_chunks(r_ds, nchunks=1)
        r_chunk_list = r_fmridataset.collect_chunks(r_chunks)
        r_data = np.array(r_chunk_list[0].rx2("data")).reshape(20, 10, order="F")

        np.testing.assert_array_almost_equal(py_chunk.data, r_data)


def test_runwise_chunk_count(r_fmridataset) -> None:
    """Runwise chunking produces the same number of chunks."""
    from rpy2.robjects import r as R
    from rpy2.robjects import conversion, default_converter

    with conversion.localconverter(default_converter):
        mat = np.zeros((30, 5))

        # Python
        py_ds = matrix_dataset(mat, TR=1.0, run_length=[10, 10, 10])
        py_chunks = data_chunks(py_ds, runwise=True).collect()

        # R
        r_mat = R.matrix(mat.ravel(order="F").tolist(), nrow=30, ncol=5)
        r_ds = r_fmridataset.matrix_dataset(r_mat, TR=1.0, run_length=R.c(10, 10, 10))
        r_chunks = r_fmridataset.data_chunks(r_ds, runwise=True)
        r_chunk_list = r_fmridataset.collect_chunks(r_chunks)

        assert len(py_chunks) == len(r_chunk_list)
