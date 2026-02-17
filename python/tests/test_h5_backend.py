"""Tests for H5Backend (HDF5 storage via h5py)."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from fmridataset.backends.h5_backend import H5Backend
from fmridataset.errors import BackendIOError


@pytest.fixture()
def h5_file(tmp_path, rng):
    """Create a small HDF5 file in fmristore layout."""
    nx, ny, nz, nt = 3, 3, 3, 10
    data_4d = rng.standard_normal((nx, ny, nz, nt)).astype(np.float64)
    mask_3d = np.ones((nx, ny, nz), dtype=np.bool_)
    mask_3d[0, 0, 0] = False  # one voxel excluded

    # Data file: "data" is a dataset (4D array)
    path = tmp_path / "test.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("data", data=data_4d)
        f.attrs["dim"] = [nx, ny, nz, nt]

    # Mask file: "data" is a group, "data/elements" is indices dataset
    mask_path = tmp_path / "mask.h5"
    elements = np.where(mask_3d.ravel())[0].astype(np.int64)
    with h5py.File(str(mask_path), "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("elements", data=elements)

    return path, mask_path, data_4d, mask_3d


class TestH5BackendCreation:
    def test_open_and_dims(self, h5_file):
        path, mask_path, data_4d, mask_3d = h5_file
        backend = H5Backend(source=str(path), mask_source=str(mask_path))
        backend.open()
        dims = backend.get_dims()
        assert dims.spatial == (3, 3, 3)
        assert dims.time == 10
        backend.close()

    def test_context_manager(self, h5_file):
        path, mask_path, _, _ = h5_file
        with H5Backend(source=str(path), mask_source=str(mask_path)) as b:
            assert b.get_dims().time == 10

    def test_missing_file_raises(self, tmp_path, h5_file):
        _, mask_path, _, _ = h5_file
        backend = H5Backend(
            source=str(tmp_path / "nonexistent.h5"),
            mask_source=str(mask_path),
        )
        with pytest.raises(BackendIOError):
            backend.open()

    def test_missing_mask_raises(self, h5_file, tmp_path):
        path, _, _, _ = h5_file
        backend = H5Backend(
            source=str(path),
            mask_source=str(tmp_path / "nonexistent_mask.h5"),
        )
        with pytest.raises(BackendIOError):
            backend.open()


class TestH5BackendData:
    def test_get_all_data(self, h5_file):
        path, mask_path, data_4d, mask_3d = h5_file
        backend = H5Backend(source=str(path), mask_source=str(mask_path))
        backend.open()
        data = backend.get_data()
        n_vox = int(mask_3d.sum())
        assert data.shape == (10, n_vox)
        backend.close()

    def test_get_data_subset_rows(self, h5_file):
        path, mask_path, _, mask_3d = h5_file
        backend = H5Backend(source=str(path), mask_source=str(mask_path))
        backend.open()
        rows = np.array([0, 5], dtype=np.intp)
        data = backend.get_data(rows=rows)
        assert data.shape[0] == 2
        assert data.shape[1] == int(mask_3d.sum())
        backend.close()

    def test_get_data_subset_cols(self, h5_file):
        path, mask_path, _, _ = h5_file
        backend = H5Backend(source=str(path), mask_source=str(mask_path))
        backend.open()
        cols = np.array([0, 1, 2], dtype=np.intp)
        data = backend.get_data(cols=cols)
        assert data.shape == (10, 3)
        backend.close()


class TestH5BackendMask:
    def test_mask_shape(self, h5_file):
        path, mask_path, _, mask_3d = h5_file
        backend = H5Backend(source=str(path), mask_source=str(mask_path))
        backend.open()
        mask = backend.get_mask()
        n_total = np.prod(mask_3d.shape)
        # mask_vec is derived from elements indices
        assert mask.dtype == np.bool_
        backend.close()


class TestH5BackendMetadata:
    def test_metadata(self, h5_file):
        path, mask_path, _, _ = h5_file
        backend = H5Backend(source=str(path), mask_source=str(mask_path))
        backend.open()
        meta = backend.get_metadata()
        assert meta["format"] == "h5"
        backend.close()
