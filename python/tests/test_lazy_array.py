"""Tests for lazy dask array conversion and series() alias."""

import warnings

import numpy as np
import pytest

from fmridataset import matrix_dataset, series


class TestSeriesAlias:
    def test_deprecated_warning(self) -> None:
        mat = np.arange(20, dtype=np.float64).reshape(4, 5)
        ds = matrix_dataset(mat, TR=1.0, run_length=4)
        with pytest.warns(DeprecationWarning, match="series.*deprecated"):
            fs = series(ds)
        assert fs.shape == (4, 5)


class TestAsDaskArray:
    @pytest.fixture()
    def simple_ds(self):
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((10, 5)).astype(np.float64)
        return matrix_dataset(mat, TR=2.0, run_length=10), mat

    def test_dask_available(self, simple_ds) -> None:
        da = pytest.importorskip("dask.array")
        from fmridataset import as_dask_array

        ds, mat = simple_ds
        lazy = as_dask_array(ds._backend)
        assert lazy.shape == (10, 5)
        result = lazy.compute()
        np.testing.assert_array_almost_equal(result, mat)

    def test_dask_dataset_wrapper(self, simple_ds) -> None:
        pytest.importorskip("dask.array")
        from fmridataset import as_dask_array_dataset

        ds, mat = simple_ds
        lazy = as_dask_array_dataset(ds)
        assert lazy.shape == (10, 5)
        result = lazy.compute()
        np.testing.assert_array_almost_equal(result, mat)

    def test_dask_slicing(self, simple_ds) -> None:
        da = pytest.importorskip("dask.array")
        from fmridataset import as_dask_array

        ds, mat = simple_ds
        lazy = as_dask_array(ds._backend)
        # Slice first 3 rows, columns 1 and 3
        sub = lazy[:3, [1, 3]]
        result = sub.compute()
        np.testing.assert_array_almost_equal(result, mat[:3, [1, 3]])
