"""Tests for FmriDataset and MatrixDataset."""

import numpy as np
import pandas as pd
import pytest

from fmridataset import (
    ConfigError,
    FmriDataset,
    MatrixBackend,
    MatrixDataset,
    fmri_zarr_dataset,
    SamplingFrame,
    fmri_dataset,
    matrix_dataset,
)


class TestMatrixDatasetConstructor:
    def test_basic(self) -> None:
        mat = np.random.default_rng(0).standard_normal((100, 50))
        ds = matrix_dataset(mat, TR=2.0, run_length=100)
        assert ds.n_timepoints == 100
        assert ds.n_runs == 1
        assert ds.TR == 2.0

    def test_multi_run(self) -> None:
        mat = np.zeros((330, 50))
        ds = matrix_dataset(mat, TR=2.0, run_length=[100, 120, 110])
        assert ds.n_runs == 3
        assert ds.blocklens == (100, 120, 110)

    def test_run_length_mismatch(self) -> None:
        mat = np.zeros((100, 50))
        with pytest.raises(ValueError, match="sum"):
            matrix_dataset(mat, TR=2.0, run_length=[50, 60])

    def test_run_length_non_integer_rejected(self) -> None:
        mat = np.zeros((5, 4))
        with pytest.raises(ValueError, match="run_length values must be integers"):
            matrix_dataset(mat, TR=2.0, run_length=[2.5, 2.5])

    def test_run_length_string_rejected(self) -> None:
        mat = np.zeros((2, 4))
        with pytest.raises(ValueError, match="run_length values must be numeric"):
            matrix_dataset(mat, TR=2.0, run_length="2")

    def test_vector_input_becomes_column_matrix(self) -> None:
        vec = np.arange(6, dtype=float)
        ds = matrix_dataset(vec, TR=1.0, run_length=6)
        assert ds.datamat.shape == (6, 1)
        np.testing.assert_array_equal(ds.datamat[:, 0], vec)

    def test_event_table(self) -> None:
        mat = np.zeros((100, 50))
        et = pd.DataFrame({"onset": [10, 20], "condition": ["A", "B"]})
        ds = matrix_dataset(mat, TR=2.0, run_length=100, event_table=et)
        assert len(ds.event_table) == 2

    def test_default_event_table_has_event_index(self) -> None:
        mat = np.zeros((20, 4))
        ds = matrix_dataset(mat, TR=2.0, run_length=20)
        assert len(ds.event_table.columns) == 0
        assert len(ds.event_table) == 0

    def test_empty_event_table_is_normalized(self) -> None:
        mat = np.zeros((20, 4))
        ds = matrix_dataset(
            mat,
            TR=2.0,
            run_length=20,
            event_table=pd.DataFrame(),
        )
        assert len(ds.event_table.columns) == 0
        assert len(ds.event_table) == 0

    def test_multi_dimensional_datamat_rejected(self) -> None:
        mat = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        with pytest.raises(ValueError, match="datamat must be a 2-D matrix"):
            matrix_dataset(
                mat,
                TR=2.0,
                run_length=24,
            )

    def test_datamat_property(self) -> None:
        mat = np.random.default_rng(1).standard_normal((10, 5))
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        np.testing.assert_array_equal(ds.datamat, mat)


class TestZarrDatasetConstructor:
    def test_fmri_zarr_dataset_alias(self, zarr_store):
        zarr_dataset = fmri_zarr_dataset(
            zarr_source=str(zarr_store),
            TR=2.0,
            run_length=10,
            preload=False,
        )

        assert isinstance(zarr_dataset, FmriDataset)
        assert zarr_dataset.n_timepoints == 10


class TestFmriDatasetFromBackend:
    def test_basic(self) -> None:
        mat = np.zeros((20, 10))
        backend = MatrixBackend(data_matrix=mat)
        ds = fmri_dataset(backend, TR=1.0, run_length=[10, 10])
        assert ds.n_timepoints == 20
        assert ds.n_runs == 2

    def test_time_mismatch(self) -> None:
        mat = np.zeros((20, 10))
        backend = MatrixBackend(data_matrix=mat)
        with pytest.raises(ConfigError, match="n_timepoints"):
            fmri_dataset(backend, TR=1.0, run_length=[15, 10])

    def test_run_length_non_integer_rejected(self) -> None:
        mat = np.zeros((5, 10))
        backend = MatrixBackend(data_matrix=mat)
        with pytest.raises(ValueError, match="run_length values must be integers"):
            fmri_dataset(backend, TR=1.0, run_length=[2.5, 2.5])

    def test_run_length_string_rejected(self) -> None:
        mat = np.zeros((2, 10))
        backend = MatrixBackend(data_matrix=mat)
        with pytest.raises(ValueError, match="run_length values must be numeric"):
            fmri_dataset(backend, TR=1.0, run_length="2")

    def test_default_event_table_remains_empty(self) -> None:
        mat = np.zeros((20, 10))
        backend = MatrixBackend(data_matrix=mat)
        ds = fmri_dataset(backend, TR=1.0, run_length=20)
        assert len(ds.event_table.columns) == 0
        assert len(ds.event_table) == 0

    def test_duplicate_event_table_columns_error(self) -> None:
        mat = np.zeros((20, 10))
        backend = MatrixBackend(data_matrix=mat)
        bad_events = pd.DataFrame([[1, 2], [3, 4]], columns=["onset", "onset"])
        with pytest.raises(ValueError, match="columns must be unique"):
            fmri_dataset(
                backend,
                TR=1.0,
                run_length=20,
                event_table=bad_events,
            )


class TestDataAccess:
    def test_get_data(self) -> None:
        rng = np.random.default_rng(2)
        mat = rng.standard_normal((10, 5))
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        data = ds.get_data()
        np.testing.assert_array_equal(data, mat)

    def test_get_data_matrix_rows(self) -> None:
        mat = np.arange(50, dtype=float).reshape(10, 5)
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        sub = ds.get_data_matrix(rows=np.array([0, 2]))
        assert sub.shape == (2, 5)
        np.testing.assert_array_equal(sub[0], mat[0])
        np.testing.assert_array_equal(sub[1], mat[2])

    def test_get_mask(self) -> None:
        mat = np.zeros((10, 5))
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        mask = ds.get_mask()
        assert mask.shape == (5,)
        assert mask.all()

    def test_get_dims(self) -> None:
        mat = np.zeros((10, 5))
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        dims = ds.get_dims()
        assert dims.time == 10
        assert dims.spatial == (5, 1, 1)


class TestBlockids:
    def test_blockids_two_runs(self) -> None:
        mat = np.zeros((7, 3))
        ds = matrix_dataset(mat, TR=1.0, run_length=[3, 4])
        expected = np.array([1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(ds.blockids, expected)


class TestRepr:
    def test_matrix_dataset_repr(self) -> None:
        mat = np.zeros((10, 5))
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        r = repr(ds)
        assert "MatrixDataset" in r
        assert "10" in r
        assert "5" in r

    def test_fmri_dataset_repr(self) -> None:
        mat = np.zeros((10, 5))
        backend = MatrixBackend(data_matrix=mat)
        ds = fmri_dataset(backend, TR=1.0, run_length=10)
        r = repr(ds)
        assert "FmriDataset" in r
