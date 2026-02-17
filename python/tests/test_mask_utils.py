"""Tests for mask_utils — mask_to_logical / mask_to_volume."""

import numpy as np
import pytest

from fmridataset.mask_utils import mask_to_logical, mask_to_volume


class TestMaskToLogical:
    def test_bool_passthrough(self) -> None:
        mask = np.array([True, False, True])
        result = mask_to_logical(mask)
        np.testing.assert_array_equal(result, [True, False, True])
        assert result.dtype == np.bool_

    def test_numeric_to_bool(self) -> None:
        mask = np.array([1, 0, 1, 0, 1])
        result = mask_to_logical(mask)
        np.testing.assert_array_equal(result, [True, False, True, False, True])

    def test_3d_array_flattened(self) -> None:
        mask = np.zeros((2, 2, 2), dtype=np.bool_)
        mask[0, 0, 0] = True
        mask[1, 1, 1] = True
        result = mask_to_logical(mask)
        assert result.ndim == 1
        assert result.shape == (8,)
        assert result.sum() == 2

    def test_float_to_bool(self) -> None:
        mask = np.array([0.0, 1.0, 0.5])
        result = mask_to_logical(mask)
        np.testing.assert_array_equal(result, [False, True, True])


class TestMaskToVolume:
    def test_basic_reshape(self) -> None:
        vec = np.array([True, False, True, False, True, False, True, True],
                       dtype=np.bool_)
        vol = mask_to_volume(vec, (2, 2, 2))
        assert vol.shape == (2, 2, 2)
        assert vol.dtype == np.bool_
        assert vol.ravel().sum() == 5

    def test_roundtrip(self) -> None:
        dims = (3, 4, 5)
        rng = np.random.default_rng(42)
        original = rng.random(dims) > 0.5
        flat = original.ravel()
        recovered = mask_to_volume(flat, dims)
        np.testing.assert_array_equal(recovered, original)

    def test_size_mismatch_raises(self) -> None:
        vec = np.ones(10, dtype=np.bool_)
        with pytest.raises(ValueError, match="doesn't match"):
            mask_to_volume(vec, (2, 2, 2))
