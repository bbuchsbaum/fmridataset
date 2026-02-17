"""Tests for series selectors."""

import numpy as np
import pytest

from fmridataset import (
    AllSelector,
    IndexSelector,
    ROISelector,
    matrix_dataset,
)


@pytest.fixture()
def dataset():
    mat = np.zeros((10, 20))
    return matrix_dataset(mat, TR=1.0, run_length=10)


class TestIndexSelector:
    def test_basic(self, dataset) -> None:
        sel = IndexSelector([0, 5, 10])
        idx = sel.resolve_indices(dataset)
        np.testing.assert_array_equal(idx, [0, 5, 10])

    def test_out_of_range(self, dataset) -> None:
        sel = IndexSelector([0, 25])
        with pytest.raises(IndexError):
            sel.resolve_indices(dataset)


class TestAllSelector:
    def test_basic(self, dataset) -> None:
        sel = AllSelector()
        idx = sel.resolve_indices(dataset)
        np.testing.assert_array_equal(idx, np.arange(20))


class TestROISelector:
    def test_basic(self, dataset) -> None:
        roi = np.zeros(20, dtype=bool)
        roi[5] = True
        roi[10] = True
        sel = ROISelector(roi)
        idx = sel.resolve_indices(dataset)
        np.testing.assert_array_equal(idx, [5, 10])

    def test_size_mismatch(self, dataset) -> None:
        sel = ROISelector(np.ones(15, dtype=bool))
        with pytest.raises(ValueError, match="ROI mask length"):
            sel.resolve_indices(dataset)
