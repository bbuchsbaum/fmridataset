"""Tests for LatentBackend (latent-decomposition HDF5 files)."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from fmridataset.backends.latent_backend import LatentBackend
from fmridataset.errors import BackendIOError


@pytest.fixture()
def latent_h5(tmp_path, rng):
    """Create a small latent-decomposition HDF5 file."""
    n_time, n_components, n_voxels = 20, 5, 100
    basis = rng.standard_normal((n_time, n_components)).astype(np.float64)
    loadings = rng.standard_normal((n_voxels, n_components)).astype(np.float64)
    offset = rng.standard_normal(n_voxels).astype(np.float64)

    path = tmp_path / "test.lv.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("basis", data=basis)
        f.create_dataset("loadings", data=loadings)
        f.create_dataset("offset", data=offset)

    expected = basis @ loadings.T + offset[np.newaxis, :]
    return path, basis, loadings, offset, expected


@pytest.fixture()
def latent_h5_no_offset(tmp_path, rng):
    """Latent HDF5 without offset."""
    n_time, n_components, n_voxels = 15, 3, 50
    basis = rng.standard_normal((n_time, n_components)).astype(np.float64)
    loadings = rng.standard_normal((n_voxels, n_components)).astype(np.float64)

    path = tmp_path / "test_nooff.lv.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("basis", data=basis)
        f.create_dataset("loadings", data=loadings)

    expected = basis @ loadings.T
    return path, expected


class TestLatentBackendCreation:
    def test_open_and_dims(self, latent_h5):
        path, basis, loadings, _, _ = latent_h5
        backend = LatentBackend(source=str(path))
        backend.open()
        dims = backend.get_dims()
        assert dims.time == 20
        assert dims.spatial == (100, 1, 1)
        backend.close()

    def test_missing_file_raises(self, tmp_path):
        backend = LatentBackend(source=str(tmp_path / "missing.h5"))
        with pytest.raises(BackendIOError):
            backend.open()

    def test_missing_basis_raises(self, tmp_path):
        path = tmp_path / "no_basis.h5"
        with h5py.File(str(path), "w") as f:
            f.create_dataset("loadings", data=np.zeros((10, 3)))
        backend = LatentBackend(source=str(path))
        with pytest.raises(BackendIOError, match="basis"):
            backend.open()


class TestLatentBackendReconstruction:
    def test_full_reconstruction(self, latent_h5):
        path, _, _, _, expected = latent_h5
        backend = LatentBackend(source=str(path))
        backend.open()
        data = backend.get_data()
        np.testing.assert_array_almost_equal(data, expected)
        backend.close()

    def test_reconstruction_no_offset(self, latent_h5_no_offset):
        path, expected = latent_h5_no_offset
        backend = LatentBackend(source=str(path))
        backend.open()
        data = backend.get_data()
        np.testing.assert_array_almost_equal(data, expected)
        backend.close()

    def test_preload(self, latent_h5):
        path, _, _, _, expected = latent_h5
        backend = LatentBackend(source=str(path), preload=True)
        backend.open()
        data = backend.get_data()
        np.testing.assert_array_almost_equal(data, expected)
        backend.close()

    def test_subset_rows(self, latent_h5):
        path, _, _, _, expected = latent_h5
        backend = LatentBackend(source=str(path))
        backend.open()
        rows = np.array([0, 5, 10], dtype=np.intp)
        data = backend.get_data(rows=rows)
        np.testing.assert_array_almost_equal(data, expected[rows])
        backend.close()

    def test_subset_cols(self, latent_h5):
        path, _, _, _, expected = latent_h5
        backend = LatentBackend(source=str(path))
        backend.open()
        cols = np.array([0, 10, 20], dtype=np.intp)
        data = backend.get_data(cols=cols)
        np.testing.assert_array_almost_equal(data, expected[:, cols])
        backend.close()


class TestLatentBackendMultiFile:
    def test_two_files(self, tmp_path, rng):
        n_comp, n_vox = 4, 30
        loadings = rng.standard_normal((n_vox, n_comp)).astype(np.float64)

        basis1 = rng.standard_normal((10, n_comp)).astype(np.float64)
        basis2 = rng.standard_normal((8, n_comp)).astype(np.float64)

        p1 = tmp_path / "run1.lv.h5"
        p2 = tmp_path / "run2.lv.h5"
        with h5py.File(str(p1), "w") as f:
            f.create_dataset("basis", data=basis1)
            f.create_dataset("loadings", data=loadings)
        with h5py.File(str(p2), "w") as f:
            f.create_dataset("basis", data=basis2)

        backend = LatentBackend(source=[str(p1), str(p2)])
        backend.open()
        dims = backend.get_dims()
        assert dims.time == 18  # 10 + 8

        full_basis = np.concatenate([basis1, basis2], axis=0)
        expected = full_basis @ loadings.T
        data = backend.get_data()
        np.testing.assert_array_almost_equal(data, expected)
        backend.close()


class TestLatentBackendMask:
    def test_all_voxels_valid(self, latent_h5):
        path, _, loadings, _, _ = latent_h5
        backend = LatentBackend(source=str(path))
        backend.open()
        mask = backend.get_mask()
        assert mask.shape == (loadings.shape[0],)
        assert mask.all()
        backend.close()


class TestLatentBackendMetadata:
    def test_metadata(self, latent_h5):
        path, _, _, offset, _ = latent_h5
        backend = LatentBackend(source=str(path))
        backend.open()
        meta = backend.get_metadata()
        assert meta["format"] == "latent_h5"
        assert meta["n_components"] == 5
        assert meta["has_offset"] is True
        assert "basis_variance" in meta
        assert "loadings_norm" in meta
        assert "loadings_sparsity" in meta
        np.testing.assert_allclose(meta["basis_variance"], np.var(latent_h5[1], axis=0, ddof=1))
        np.testing.assert_allclose(meta["loadings_norm"], np.sqrt(np.sum(latent_h5[2] ** 2, axis=0)))
        assert meta["loadings_sparsity"] == 0.0
        backend.close()
