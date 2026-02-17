"""Tests for LatentDataset and latent_dataset() constructor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from fmridataset.latent_dataset import LatentDataset, latent_dataset


@pytest.fixture()
def latent_h5_file(tmp_path, rng):
    """Create a latent HDF5 file for testing."""
    n_time, n_comp, n_vox = 20, 4, 60
    basis = rng.standard_normal((n_time, n_comp)).astype(np.float64)
    loadings = rng.standard_normal((n_vox, n_comp)).astype(np.float64)
    offset = rng.standard_normal(n_vox).astype(np.float64)

    path = tmp_path / "test.lv.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("basis", data=basis)
        f.create_dataset("loadings", data=loadings)
        f.create_dataset("offset", data=offset)

    expected = basis @ loadings.T + offset[np.newaxis, :]
    return str(path), basis, loadings, offset, expected


class TestLatentDatasetConstructor:
    def test_basic_creation(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        assert isinstance(ds, LatentDataset)
        assert ds.n_timepoints == 20
        assert ds.TR == 2.0
        assert ds.n_runs == 1

    def test_multiple_runs(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=[10, 10])
        assert ds.n_runs == 2
        assert list(ds.blocklens) == [10, 10]

    def test_with_event_table(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        events = pd.DataFrame({"onset": [0.0, 2.0], "condition": ["A", "B"]})
        ds = latent_dataset(source=path, TR=2.0, run_length=20, event_table=events)
        assert len(ds.event_table) == 2

    def test_run_length_non_integer_rejected(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        with pytest.raises(ValueError, match="run_length values must be integers"):
            latent_dataset(source=path, TR=2.0, run_length=[2.5, 2.5])


class TestLatentDatasetMethods:
    def test_get_latent_scores(self, latent_h5_file):
        path, basis, loadings, _, expected = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        # get_latent_scores returns the reconstructed data (via get_data)
        scores = ds.get_latent_scores()
        np.testing.assert_array_almost_equal(scores, expected)

    def test_get_spatial_loadings(self, latent_h5_file):
        path, _, loadings, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        result = ds.get_spatial_loadings()
        np.testing.assert_array_almost_equal(result, loadings)

    def test_get_data(self, latent_h5_file):
        path, _, _, _, expected = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        data = ds.get_data()
        np.testing.assert_array_almost_equal(data, expected)


class TestLatentDatasetRepr:
    def test_repr(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        r = repr(ds)
        assert "LatentDataset" in r
        assert "components=4" in r
