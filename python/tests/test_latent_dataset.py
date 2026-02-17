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

    def test_duplicate_event_table_columns_error(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        bad_events = pd.DataFrame(
            [[0.0, 1.0], [2.0, 3.0]],
            columns=["onset", "onset"],
        )
        with pytest.raises(ValueError, match="columns must be unique"):
            latent_dataset(
                source=path,
                TR=2.0,
                run_length=20,
                event_table=bad_events,
            )

    def test_run_length_non_integer_rejected(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        with pytest.raises(ValueError, match="run_length values must be integers"):
            latent_dataset(source=path, TR=2.0, run_length=[2.5, 2.5])

    def test_run_length_sum_must_match_time(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        with pytest.raises(ValueError, match="sum"):
            latent_dataset(source=path, TR=2.0, run_length=[5, 10])

    def test_relative_source_resolves_base_path(self, tmp_path, rng) -> None:
        path = tmp_path / "relative.lv.h5"

        basis = rng.standard_normal((20, 4)).astype(np.float64)
        loadings = rng.standard_normal((60, 4)).astype(np.float64)
        offset = rng.standard_normal(60).astype(np.float64)

        with h5py.File(str(path), "w") as f:
            f.create_dataset("basis", data=basis)
            f.create_dataset("loadings", data=loadings)
            f.create_dataset("offset", data=offset)

        ds = latent_dataset(
            source="relative.lv.h5",
            TR=2.0,
            run_length=20,
            base_path=str(tmp_path),
        )
        assert ds.n_timepoints == 20

    def test_run_length_inferred_from_backend_when_missing(self, tmp_path, rng) -> None:
        path = tmp_path / "inferred.lv.h5"

        basis = rng.standard_normal((12, 3)).astype(np.float64)
        loadings = rng.standard_normal((25, 3)).astype(np.float64)
        offset = rng.standard_normal(25).astype(np.float64)

        with h5py.File(str(path), "w") as f:
            f.create_dataset("basis", data=basis)
            f.create_dataset("loadings", data=loadings)
            f.create_dataset("offset", data=offset)

        ds = latent_dataset(source=str(path), TR=2.0)
        assert ds.n_timepoints == 12
        assert list(ds.blocklens) == [12]

    def test_run_length_inferred_from_run_length_alias_zero(self, tmp_path, rng) -> None:
        run_lengths = [7, 13]
        loadings = rng.standard_normal((18, 4)).astype(np.float64)

        paths = []
        for i, n_time in enumerate(run_lengths):
            path = tmp_path / f"run{i}.lv.h5"
            basis = rng.standard_normal((n_time, 4)).astype(np.float64)
            with h5py.File(str(path), "w") as f:
                f.create_dataset("basis", data=basis)
                f.create_dataset("loadings", data=loadings)
            paths.append(str(path))

        ds = latent_dataset(
            source=paths,
            TR=2.0,
            run_length=0,
        )
        assert list(ds.blocklens) == run_lengths
        assert ds.n_runs == 2
        assert ds.n_timepoints == sum(run_lengths)

    def test_run_length_inferred_from_empty_vector(self, tmp_path, rng) -> None:
        path = tmp_path / "inferred-empty.lv.h5"
        basis = rng.standard_normal((11, 3)).astype(np.float64)
        loadings = rng.standard_normal((15, 3)).astype(np.float64)
        offset = rng.standard_normal(15).astype(np.float64)

        with h5py.File(str(path), "w") as f:
            f.create_dataset("basis", data=basis)
            f.create_dataset("loadings", data=loadings)
            f.create_dataset("offset", data=offset)

        ds = latent_dataset(source=str(path), TR=2.0, run_length=[])
        assert ds.n_timepoints == 11
        assert list(ds.blocklens) == [11]

    def test_default_event_table_remains_empty(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        assert len(ds.event_table.columns) == 0
        assert len(ds.event_table) == 0

    def test_censor_propagates_to_dataset(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        censor = np.array([1, 0, 1, 0] + [0] * 16, dtype=np.intp)
        ds = latent_dataset(
            source=path,
            TR=2.0,
            run_length=20,
            censor=censor,
        )
        np.testing.assert_array_equal(ds.censor, censor)

    def test_run_length_provided_default_censor_zero_vector(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        np.testing.assert_array_equal(ds.censor, np.zeros(20, dtype=np.intp))


class TestLatentDatasetMethods:
    def test_get_latent_scores(self, latent_h5_file):
        path, basis, loadings, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        scores = ds.get_latent_scores()
        np.testing.assert_array_almost_equal(scores, basis)

    def test_get_spatial_loadings(self, latent_h5_file):
        path, _, loadings, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        result = ds.get_spatial_loadings()
        np.testing.assert_array_almost_equal(result, loadings)

    def test_get_spatial_loadings_components(self, latent_h5_file):
        path, _, loadings, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        comps = np.array([0, 2], dtype=np.intp)
        result = ds.get_spatial_loadings(components=comps)
        np.testing.assert_array_almost_equal(result, loadings[:, comps])

    def test_get_mask_is_component_mask(self, latent_h5_file):
        path, _, loadings, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        mask = ds.get_mask()
        assert mask.dtype == np.bool_
        assert mask.shape == (loadings.shape[1],)
        assert mask.all()

    def test_get_component_info(self, latent_h5_file):
        path, _, loadings, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        info = ds.get_component_info()
        assert isinstance(info, dict)
        assert info["storage_format"] == "latent"
        assert info["n_components"] == loadings.shape[1]
        assert info["format"] == "latent_h5"

    def test_get_component_info_metadata_fields(self, latent_h5_file):
        path, basis, loadings, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        info = ds.get_component_info()

        np.testing.assert_allclose(
            info["basis_variance"],
            np.var(basis, axis=0, ddof=1),
        )
        np.testing.assert_allclose(
            info["loadings_norm"],
            np.sqrt(np.sum(loadings**2, axis=0)),
        )
        assert info["loadings_sparsity"] == 0
        assert info["n_voxels"] == loadings.shape[0]
        assert info["n_runs"] == 1

    def test_get_data(self, latent_h5_file):
        path, basis, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        data = ds.get_data()
        np.testing.assert_array_almost_equal(data, basis)

    def test_get_data_warns(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        with pytest.warns(UserWarning, match="latent scores"):
            _ = ds.get_data()

    def test_reconstruct_voxels(self, latent_h5_file):
        path, _, _, _, expected = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        recon = ds.reconstruct_voxels()
        np.testing.assert_array_almost_equal(recon, expected)

        row_idx = np.array([0, 3, 5], dtype=np.intp)
        vox_idx = np.array([0, 2], dtype=np.intp)
        recon_sub = ds.reconstruct_voxels(rows=row_idx, voxels=vox_idx)
        np.testing.assert_array_almost_equal(recon_sub, expected[row_idx][:, vox_idx])


class TestLatentDatasetRepr:
    def test_repr(self, latent_h5_file):
        path, _, _, _, _ = latent_h5_file
        ds = latent_dataset(source=path, TR=2.0, run_length=20)
        r = repr(ds)
        assert "LatentDataset" in r
        assert "components=4" in r
