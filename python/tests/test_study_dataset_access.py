"""Tests for StudyDataset subject-level data access."""

import numpy as np
import pytest

from fmridataset import matrix_dataset, study_dataset
from fmridataset.errors import ConfigError


@pytest.fixture()
def two_subject_study():
    rng = np.random.default_rng(99)
    mat1 = rng.standard_normal((20, 10)).astype(np.float64)
    mat2 = rng.standard_normal((20, 10)).astype(np.float64)
    ds1 = matrix_dataset(mat1, TR=2.0, run_length=20)
    ds2 = matrix_dataset(mat2, TR=2.0, run_length=20)
    sds = study_dataset([ds1, ds2], subject_ids=["S01", "S02"])
    return sds, mat1, mat2


class TestStudyDatasetSubjectAccess:
    def test_get_data_matrix_all(self, two_subject_study) -> None:
        sds, mat1, mat2 = two_subject_study
        full = sds.get_data_matrix()
        assert full.shape == (40, 10)

    def test_get_subject_data_single(self, two_subject_study) -> None:
        sds, mat1, mat2 = two_subject_study
        s1_data = sds.get_subject_data(subject_id="S01")
        assert s1_data.shape == (20, 10)
        np.testing.assert_array_almost_equal(s1_data, mat1)

    def test_get_subject_data_second(self, two_subject_study) -> None:
        sds, mat1, mat2 = two_subject_study
        s2_data = sds.get_subject_data(subject_id="S02")
        assert s2_data.shape == (20, 10)
        np.testing.assert_array_almost_equal(s2_data, mat2)

    def test_get_subject_data_multiple(self, two_subject_study) -> None:
        sds, mat1, mat2 = two_subject_study
        both = sds.get_subject_data(subject_id=["S01", "S02"])
        assert both.shape == (40, 10)
        np.testing.assert_array_almost_equal(both[:20], mat1)
        np.testing.assert_array_almost_equal(both[20:], mat2)

    def test_get_subject_data_invalid(self, two_subject_study) -> None:
        sds, _, _ = two_subject_study
        with pytest.raises(ConfigError, match="not found"):
            sds.get_subject_data(subject_id="INVALID")

    def test_get_subject_data_with_cols(self, two_subject_study) -> None:
        sds, mat1, _ = two_subject_study
        cols = np.array([0, 3, 7], dtype=np.intp)
        s1_sub = sds.get_subject_data(subject_id="S01", cols=cols)
        assert s1_sub.shape == (20, 3)
        np.testing.assert_array_almost_equal(s1_sub, mat1[:, cols])
