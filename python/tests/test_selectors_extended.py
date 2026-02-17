"""Tests for VoxelSelector, SphereSelector, MaskSelector."""

import numpy as np
import pytest

from fmridataset import matrix_dataset
from fmridataset.selectors import (
    MaskSelector,
    SphereSelector,
    VoxelSelector,
)


@pytest.fixture()
def simple_ds():
    """3x3x1 spatial, 5 timepoints, all voxels in mask."""
    mat = np.arange(45, dtype=np.float64).reshape(5, 9)
    return matrix_dataset(mat, TR=2.0, run_length=5)


class TestVoxelSelector:
    def test_single_voxel(self, simple_ds) -> None:
        # (1,1,1) is 0-based linear index 0 -> masked col 0
        sel = VoxelSelector(np.array([1, 1, 1]))
        idx = sel.resolve_indices(simple_ds)
        assert idx.shape == (1,)
        assert idx[0] == 0

    def test_multiple_voxels(self, simple_ds) -> None:
        coords = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]])
        sel = VoxelSelector(coords)
        idx = sel.resolve_indices(simple_ds)
        np.testing.assert_array_equal(idx, [0, 1, 2])

    def test_out_of_range(self, simple_ds) -> None:
        sel = VoxelSelector(np.array([10, 1, 1]))
        with pytest.raises(IndexError):
            sel.resolve_indices(simple_ds)

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            VoxelSelector(np.array([1, 2]))

    def test_2d_bad_cols(self) -> None:
        with pytest.raises(ValueError, match="shape \\(N, 3\\)"):
            VoxelSelector(np.array([[1, 2], [3, 4]]))


class TestSphereSelector:
    def test_sphere_center(self, simple_ds) -> None:
        # dims are (9,1,1); center at (5,1,1), radius 0.5 -> only that voxel
        sel = SphereSelector(center=[5, 1, 1], radius=0.5)
        idx = sel.resolve_indices(simple_ds)
        assert len(idx) == 1
        assert idx[0] == 4  # 0-based col index

    def test_sphere_radius_1(self, simple_ds) -> None:
        # dims (9,1,1); center (5,1,1), radius 1.1 -> cols 3,4,5
        sel = SphereSelector(center=[5, 1, 1], radius=1.1)
        idx = sel.resolve_indices(simple_ds)
        assert len(idx) == 3
        np.testing.assert_array_equal(idx, [3, 4, 5])

    def test_negative_radius(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SphereSelector(center=[1, 1, 1], radius=-1)

    def test_no_overlap(self) -> None:
        # 2x2x1 dataset, sphere far away
        mat = np.zeros((5, 4), dtype=np.float64)
        ds = matrix_dataset(mat, TR=1.0, run_length=5)
        sel = SphereSelector(center=[100, 100, 100], radius=1.0)
        with pytest.raises(ValueError, match="does not overlap"):
            sel.resolve_indices(ds)


class TestMaskSelector:
    def test_full_volume_mask(self, simple_ds) -> None:
        # Select first 3 voxels out of 9
        mask = np.zeros(9, dtype=bool)
        mask[:3] = True
        sel = MaskSelector(mask)
        idx = sel.resolve_indices(simple_ds)
        np.testing.assert_array_equal(idx, [0, 1, 2])

    def test_masked_space_mask(self, simple_ds) -> None:
        # All 9 voxels are in mask; select last 2
        mask = np.zeros(9, dtype=bool)
        mask[7:] = True
        sel = MaskSelector(mask)
        idx = sel.resolve_indices(simple_ds)
        np.testing.assert_array_equal(idx, [7, 8])

    def test_wrong_length(self, simple_ds) -> None:
        mask = np.ones(5, dtype=bool)
        sel = MaskSelector(mask)
        with pytest.raises(ValueError, match="does not match"):
            sel.resolve_indices(simple_ds)

    def test_no_voxels(self, simple_ds) -> None:
        mask = np.zeros(9, dtype=bool)
        sel = MaskSelector(mask)
        with pytest.raises(ValueError, match="no voxels"):
            sel.resolve_indices(simple_ds)
