"""Tests for ZarrBackend."""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from fmridataset.backends.zarr_backend import ZarrBackend
from fmridataset.errors import BackendIOError


@pytest.fixture()
def zarr_store(tmp_path, rng):
    """Create a small 4D zarr array on disk."""
    nx, ny, nz, nt = 3, 3, 3, 10
    data_4d = rng.standard_normal((nx, ny, nz, nt)).astype(np.float64)
    store_path = str(tmp_path / "test.zarr")
    z = zarr.open(store_path, mode="w", shape=(nx, ny, nz, nt), dtype=np.float64)
    z[:] = data_4d
    return store_path, data_4d


class TestZarrBackendCreation:
    def test_open_and_dims(self, zarr_store):
        path, data_4d = zarr_store
        backend = ZarrBackend(source=path)
        backend.open()
        dims = backend.get_dims()
        assert dims.spatial == (3, 3, 3)
        assert dims.time == 10
        backend.close()

    def test_context_manager(self, zarr_store):
        path, _ = zarr_store
        with ZarrBackend(source=path) as b:
            assert b.get_dims().time == 10

    def test_missing_store_raises(self, tmp_path):
        backend = ZarrBackend(source=str(tmp_path / "nonexistent.zarr"))
        with pytest.raises((BackendIOError, Exception)):
            backend.open()


class TestZarrBackendData:
    def test_get_all_data(self, zarr_store):
        path, data_4d = zarr_store
        backend = ZarrBackend(source=path)
        backend.open()
        data = backend.get_data()
        n_vox = np.prod(data_4d.shape[:3])
        assert data.shape == (10, n_vox)
        backend.close()

    def test_get_data_subset_rows(self, zarr_store):
        path, _ = zarr_store
        backend = ZarrBackend(source=path)
        backend.open()
        rows = np.array([0, 5], dtype=np.intp)
        data = backend.get_data(rows=rows)
        assert data.shape[0] == 2
        backend.close()

    def test_get_data_subset_cols(self, zarr_store):
        path, _ = zarr_store
        backend = ZarrBackend(source=path)
        backend.open()
        cols = np.array([0, 1, 2], dtype=np.intp)
        data = backend.get_data(cols=cols)
        assert data.shape == (10, 3)
        backend.close()

    def test_data_values_match(self, zarr_store):
        path, data_4d = zarr_store
        backend = ZarrBackend(source=path)
        backend.open()
        data = backend.get_data()
        # Reshape 4D to (time, voxels) for comparison
        nx, ny, nz, nt = data_4d.shape
        expected = data_4d.reshape(-1, nt).T  # (nt, nx*ny*nz)
        np.testing.assert_array_almost_equal(data, expected)
        backend.close()


class TestZarrBackendMask:
    def test_default_mask_all_true(self, zarr_store):
        path, data_4d = zarr_store
        backend = ZarrBackend(source=path)
        backend.open()
        mask = backend.get_mask()
        assert mask.all()
        assert mask.shape == (np.prod(data_4d.shape[:3]),)
        backend.close()

    def test_external_mask(self, zarr_store):
        path, data_4d = zarr_store
        mask_3d = np.ones(data_4d.shape[:3], dtype=np.bool_)
        mask_3d[0, 0, 0] = False
        backend = ZarrBackend(source=path, mask=mask_3d)
        backend.open()
        mask = backend.get_mask()
        assert mask.sum() == mask_3d.sum()
        backend.close()


class TestZarrBackendMetadata:
    def test_metadata(self, zarr_store):
        path, _ = zarr_store
        backend = ZarrBackend(source=path)
        backend.open()
        meta = backend.get_metadata()
        assert meta["storage_format"] == "zarr"
        backend.close()
