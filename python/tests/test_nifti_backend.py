"""Tests for NiftiBackend (NIfTI file access via nibabel)."""

from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from fmridataset.backends.nifti_backend import NiftiBackend
from fmridataset.errors import BackendIOError


@pytest.fixture()
def nifti_with_mask(tmp_path, rng):
    """Create a 4D NIfTI with a separate 3D mask."""
    nx, ny, nz, nt = 3, 3, 3, 10
    data_4d = rng.standard_normal((nx, ny, nz, nt)).astype(np.float32)
    mask_3d = np.ones((nx, ny, nz), dtype=np.uint8)
    mask_3d[0, 0, 0] = 0  # exclude one voxel

    affine = np.eye(4)
    data_path = tmp_path / "data.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data_4d, affine), str(data_path))
    nib.save(nib.Nifti1Image(mask_3d, affine), str(mask_path))
    return str(data_path), str(mask_path), data_4d, mask_3d.astype(bool)


@pytest.fixture()
def nifti_full_mask(tmp_path, rng):
    """Create a 4D NIfTI with all-ones mask (all voxels included)."""
    nx, ny, nz, nt = 3, 3, 3, 10
    data_4d = rng.standard_normal((nx, ny, nz, nt)).astype(np.float32)
    mask_3d = np.ones((nx, ny, nz), dtype=np.uint8)

    affine = np.eye(4)
    data_path = tmp_path / "data_full.nii.gz"
    mask_path = tmp_path / "mask_full.nii.gz"
    nib.save(nib.Nifti1Image(data_4d, affine), str(data_path))
    nib.save(nib.Nifti1Image(mask_3d, affine), str(mask_path))
    return str(data_path), str(mask_path), data_4d


class TestNiftiBackendCreation:
    def test_open_and_dims(self, nifti_full_mask):
        data_path, mask_path, data_4d = nifti_full_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        dims = backend.get_dims()
        assert dims.spatial == (3, 3, 3)
        assert dims.time == 10
        backend.close()

    def test_context_manager(self, nifti_full_mask):
        data_path, mask_path, _ = nifti_full_mask
        with NiftiBackend(source=data_path, mask_source=mask_path) as b:
            assert b.get_dims().time == 10

    def test_missing_file_raises(self, tmp_path, nifti_full_mask):
        _, mask_path, _ = nifti_full_mask
        backend = NiftiBackend(
            source=str(tmp_path / "missing.nii.gz"),
            mask_source=mask_path,
        )
        with pytest.raises(BackendIOError):
            backend.open()

    def test_missing_mask_raises(self, tmp_path, nifti_full_mask):
        data_path, _, _ = nifti_full_mask
        backend = NiftiBackend(
            source=data_path,
            mask_source=str(tmp_path / "missing_mask.nii.gz"),
        )
        with pytest.raises(BackendIOError):
            backend.open()


class TestNiftiBackendData:
    def test_get_all_data(self, nifti_full_mask):
        data_path, mask_path, data_4d = nifti_full_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        data = backend.get_data()
        n_vox = np.prod(data_4d.shape[:3])
        assert data.shape == (10, n_vox)
        backend.close()

    def test_get_data_values(self, nifti_full_mask):
        data_path, mask_path, data_4d = nifti_full_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        data = backend.get_data()
        expected = data_4d.reshape(-1, data_4d.shape[-1]).T
        np.testing.assert_array_almost_equal(data, expected, decimal=5)
        backend.close()

    def test_get_data_subset_rows(self, nifti_full_mask):
        data_path, mask_path, _ = nifti_full_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        rows = np.array([0, 5], dtype=np.intp)
        data = backend.get_data(rows=rows)
        assert data.shape[0] == 2
        backend.close()

    def test_get_data_subset_cols(self, nifti_full_mask):
        data_path, mask_path, _ = nifti_full_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        cols = np.array([0, 1, 2], dtype=np.intp)
        data = backend.get_data(cols=cols)
        assert data.shape == (10, 3)
        backend.close()

    def test_get_data_with_partial_mask(self, nifti_with_mask):
        data_path, mask_path, data_4d, mask_3d = nifti_with_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        data = backend.get_data()
        n_vox = int(mask_3d.sum())
        assert data.shape == (10, n_vox)
        backend.close()


class TestNiftiBackendMask:
    def test_full_mask(self, nifti_full_mask):
        data_path, mask_path, data_4d = nifti_full_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        mask = backend.get_mask()
        assert mask.shape == (np.prod(data_4d.shape[:3]),)
        assert mask.all()
        backend.close()

    def test_partial_mask(self, nifti_with_mask):
        data_path, mask_path, _, mask_3d = nifti_with_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        mask = backend.get_mask()
        assert mask.sum() == mask_3d.sum()
        backend.close()


class TestNiftiBackendMetadata:
    def test_metadata_has_affine(self, nifti_full_mask):
        data_path, mask_path, _ = nifti_full_mask
        backend = NiftiBackend(source=data_path, mask_source=mask_path)
        backend.open()
        meta = backend.get_metadata()
        assert "affine" in meta
        assert "voxel_dims" in meta
        backend.close()
