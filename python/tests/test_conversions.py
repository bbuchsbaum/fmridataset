"""Tests for conversions."""

import numpy as np

from fmridataset import (
    MatrixBackend,
    MatrixDataset,
    fmri_dataset,
    matrix_dataset,
    to_matrix_dataset,
)


class TestToMatrixDataset:
    def test_already_matrix(self) -> None:
        mat = np.zeros((10, 5))
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        result = to_matrix_dataset(ds)
        assert result is ds  # same object

    def test_from_fmri_dataset(self) -> None:
        mat = np.random.default_rng(3).standard_normal((20, 8))
        backend = MatrixBackend(data_matrix=mat)
        ds = fmri_dataset(backend, TR=1.0, run_length=[10, 10])
        result = to_matrix_dataset(ds)
        assert isinstance(result, MatrixDataset)
        assert result.n_timepoints == 20
        assert result.n_runs == 2
        np.testing.assert_array_almost_equal(result.datamat, mat)
