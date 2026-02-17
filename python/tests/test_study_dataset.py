"""Tests for StudyDataset and study_dataset() constructor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmridataset.backends.matrix_backend import MatrixBackend
from fmridataset.dataset import FmriDataset
from fmridataset.errors import ConfigError
from fmridataset.sampling_frame import SamplingFrame
from fmridataset.study_dataset import StudyDataset, study_dataset


@pytest.fixture()
def two_datasets(rng):
    """Two simple FmriDatasets for testing study composition."""
    mat1 = rng.standard_normal((30, 50))
    mat2 = rng.standard_normal((20, 50))
    mask = np.ones(50, dtype=np.bool_)
    b1 = MatrixBackend(data_matrix=mat1, mask=mask)
    b2 = MatrixBackend(data_matrix=mat2, mask=mask)
    frame1 = SamplingFrame.create(blocklens=[15, 15], TR=2.0)
    frame2 = SamplingFrame.create(blocklens=[20], TR=2.0)
    events1 = pd.DataFrame({"onset": [0.0, 2.0], "condition": ["A", "B"]})
    events2 = pd.DataFrame({"onset": [0.0, 4.0], "condition": ["A", "C"]})
    ds1 = FmriDataset(backend=b1, sampling_frame=frame1, event_table=events1)
    ds2 = FmriDataset(backend=b2, sampling_frame=frame2, event_table=events2)
    return ds1, ds2, mat1, mat2


class TestStudyDatasetConstructor:
    def test_basic_creation(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        sds = study_dataset([ds1, ds2])
        assert isinstance(sds, StudyDataset)
        assert sds.n_timepoints == 50
        assert sds.n_runs == 3  # 2 + 1
        assert sds.TR == 2.0

    def test_custom_subject_ids(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        sds = study_dataset([ds1, ds2], subject_ids=["S01", "S02"])
        assert sds.subject_ids == ["S01", "S02"]

    def test_default_subject_ids(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        sds = study_dataset([ds1, ds2])
        assert sds.subject_ids == [1, 2]

    def test_empty_raises(self):
        with pytest.raises(ConfigError, match="non-empty"):
            study_dataset([])

    def test_mismatched_ids_raises(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        with pytest.raises(ConfigError, match="subject_ids must match"):
            study_dataset([ds1, ds2], subject_ids=["only_one"])

    def test_mismatched_tr_raises(self, rng):
        mat = rng.standard_normal((10, 20))
        mask = np.ones(20, dtype=np.bool_)
        b1 = MatrixBackend(data_matrix=mat, mask=mask)
        b2 = MatrixBackend(data_matrix=mat.copy(), mask=mask)
        ds1 = FmriDataset(
            backend=b1,
            sampling_frame=SamplingFrame.create(blocklens=[10], TR=1.0),
        )
        ds2 = FmriDataset(
            backend=b2,
            sampling_frame=SamplingFrame.create(blocklens=[10], TR=2.0),
        )
        with pytest.raises(ConfigError, match="same TR"):
            study_dataset([ds1, ds2])


class TestStudyDatasetData:
    def test_combined_data(self, two_datasets):
        ds1, ds2, mat1, mat2 = two_datasets
        sds = study_dataset([ds1, ds2])
        data = sds.get_data()
        expected = np.vstack([mat1, mat2])
        np.testing.assert_array_almost_equal(data, expected)

    def test_combined_events_have_subject_id(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        sds = study_dataset([ds1, ds2], subject_ids=["S01", "S02"])
        et = sds.event_table
        assert "subject_id" in et.columns
        assert set(et["subject_id"].unique()) == {"S01", "S02"}

    def test_combined_events_run_id_added_for_missing_run_id(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        with pytest.warns(UserWarning, match="run_id"):
            sds = study_dataset([ds1, ds2], subject_ids=["S01", "S02"])

        et = sds.event_table
        assert "run_id" in et.columns
        first_two = et.loc[et["subject_id"] == "S01", "run_id"].tolist()
        assert first_two == [1, 2]
        second = et.loc[et["subject_id"] == "S02", "run_id"].tolist()
        assert second == [1, 1]

    def test_combined_blocklens(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        sds = study_dataset([ds1, ds2])
        assert list(sds.blocklens) == [15, 15, 20]


class TestStudyDatasetRepr:
    def test_repr(self, two_datasets):
        ds1, ds2, _, _ = two_datasets
        sds = study_dataset([ds1, ds2])
        r = repr(sds)
        assert "StudyDataset" in r
        assert "subjects=2" in r
        assert "time=50" in r
