"""Tests for fmri_series() query function and resolvers."""

import numpy as np
import pytest

from fmridataset import (
    FmriSeries,
    fmri_series,
    matrix_dataset,
    resolve_selector,
    resolve_timepoints,
)
from fmridataset.selectors import AllSelector, IndexSelector, MaskSelector


@pytest.fixture()
def simple_ds():
    rng = np.random.default_rng(7)
    mat = rng.standard_normal((20, 10)).astype(np.float64)
    return matrix_dataset(mat, TR=2.0, run_length=[10, 10]), mat


class TestResolveSelector:
    def test_none_selects_all(self, simple_ds) -> None:
        ds, _ = simple_ds
        idx = resolve_selector(ds, None)
        assert len(idx) == 10

    def test_index_selector(self, simple_ds) -> None:
        ds, _ = simple_ds
        sel = IndexSelector([0, 3, 7])
        idx = resolve_selector(ds, sel)
        np.testing.assert_array_equal(idx, [0, 3, 7])

    def test_all_selector(self, simple_ds) -> None:
        ds, _ = simple_ds
        idx = resolve_selector(ds, AllSelector())
        assert len(idx) == 10

    def test_ndarray_passthrough(self, simple_ds) -> None:
        ds, _ = simple_ds
        arr = np.array([1, 5, 9], dtype=np.intp)
        idx = resolve_selector(ds, arr)
        np.testing.assert_array_equal(idx, arr)

    def test_unsupported_type(self, simple_ds) -> None:
        ds, _ = simple_ds
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_selector(ds, "bad")  # type: ignore[arg-type]


class TestResolveTimepoints:
    def test_none_selects_all(self, simple_ds) -> None:
        ds, _ = simple_ds
        idx = resolve_timepoints(ds, None)
        assert len(idx) == 20

    def test_integer_array(self, simple_ds) -> None:
        ds, _ = simple_ds
        tp = np.array([0, 5, 19], dtype=np.intp)
        idx = resolve_timepoints(ds, tp)
        np.testing.assert_array_equal(idx, tp)

    def test_boolean_mask(self, simple_ds) -> None:
        ds, _ = simple_ds
        mask = np.zeros(20, dtype=bool)
        mask[0] = True
        mask[10] = True
        idx = resolve_timepoints(ds, mask)
        np.testing.assert_array_equal(idx, [0, 10])

    def test_boolean_wrong_length(self, simple_ds) -> None:
        ds, _ = simple_ds
        mask = np.ones(5, dtype=bool)
        with pytest.raises(ValueError, match="length"):
            resolve_timepoints(ds, mask)


class TestFmriSeriesFunction:
    def test_all_data(self, simple_ds) -> None:
        ds, mat = simple_ds
        fs = fmri_series(ds)
        assert isinstance(fs, FmriSeries)
        assert fs.shape == (20, 10)
        np.testing.assert_array_almost_equal(fs.data, mat)

    def test_with_selector(self, simple_ds) -> None:
        ds, mat = simple_ds
        sel = IndexSelector([0, 2, 4])
        fs = fmri_series(ds, selector=sel)
        assert fs.shape == (20, 3)
        np.testing.assert_array_almost_equal(fs.data, mat[:, [0, 2, 4]])

    def test_with_timepoints(self, simple_ds) -> None:
        ds, mat = simple_ds
        tp = np.array([0, 1, 2], dtype=np.intp)
        fs = fmri_series(ds, timepoints=tp)
        assert fs.shape == (3, 10)
        np.testing.assert_array_almost_equal(fs.data, mat[:3])

    def test_with_both(self, simple_ds) -> None:
        ds, mat = simple_ds
        sel = IndexSelector([1, 3])
        tp = np.array([5, 6, 7], dtype=np.intp)
        fs = fmri_series(ds, selector=sel, timepoints=tp)
        assert fs.shape == (3, 2)
        np.testing.assert_array_almost_equal(fs.data, mat[5:8, [1, 3]])

    def test_temporal_info(self, simple_ds) -> None:
        ds, _ = simple_ds
        fs = fmri_series(ds)
        assert "run_id" in fs.temporal_info.columns
        assert "timepoint" in fs.temporal_info.columns
        # First 10 rows are run 1, next 10 are run 2
        assert fs.temporal_info["run_id"].iloc[0] == 1
        assert fs.temporal_info["run_id"].iloc[15] == 2

    def test_voxel_info(self, simple_ds) -> None:
        ds, _ = simple_ds
        sel = IndexSelector([3, 7])
        fs = fmri_series(ds, selector=sel)
        assert list(fs.voxel_info["voxel"]) == [3, 7]

    def test_dataset_info(self, simple_ds) -> None:
        ds, _ = simple_ds
        fs = fmri_series(ds)
        assert "backend_type" in fs.dataset_info
        assert fs.dataset_info["backend_type"] == "MatrixBackend"

    def test_mask_selector_integration(self, simple_ds) -> None:
        ds, mat = simple_ds
        mask = np.zeros(10, dtype=bool)
        mask[0] = True
        mask[9] = True
        sel = MaskSelector(mask)
        fs = fmri_series(ds, selector=sel)
        assert fs.shape == (20, 2)
        np.testing.assert_array_almost_equal(fs.data, mat[:, [0, 9]])
