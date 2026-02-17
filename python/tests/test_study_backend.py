"""Tests for StudyBackend (multi-subject composite)."""

from __future__ import annotations

import numpy as np
import pytest

from fmridataset.backends.matrix_backend import MatrixBackend
from fmridataset.backends.study_backend import StudyBackend
from fmridataset.errors import ConfigError


@pytest.fixture()
def two_backends(rng):
    """Two MatrixBackends with identical masks."""
    mat1 = rng.standard_normal((30, 50))
    mat2 = rng.standard_normal((20, 50))
    mask = np.ones(50, dtype=np.bool_)
    b1 = MatrixBackend(data_matrix=mat1, mask=mask)
    b2 = MatrixBackend(data_matrix=mat2, mask=mask)
    return b1, b2, mat1, mat2


class TestStudyBackendCreation:
    def test_basic_creation(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2])
        assert sb.get_dims().time == 50  # 30 + 20
        assert sb.get_dims().spatial == (50, 1, 1)

    def test_custom_subject_ids(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2], subject_ids=["subA", "subB"])
        assert sb.subject_ids == ["subA", "subB"]

    def test_default_subject_ids(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2])
        assert sb.subject_ids == [1, 2]

    def test_empty_backends_raises(self):
        with pytest.raises(ConfigError, match="non-empty"):
            StudyBackend(backends=[])

    def test_mismatched_subject_ids_raises(self, two_backends):
        b1, b2, _, _ = two_backends
        with pytest.raises(ConfigError, match="subject_ids must match"):
            StudyBackend(backends=[b1, b2], subject_ids=["only_one"])

    def test_mismatched_spatial_dims_raises(self, rng):
        mat1 = rng.standard_normal((10, 50))
        mat2 = rng.standard_normal((10, 30))
        b1 = MatrixBackend(data_matrix=mat1, mask=np.ones(50, dtype=np.bool_))
        b2 = MatrixBackend(data_matrix=mat2, mask=np.ones(30, dtype=np.bool_))
        with pytest.raises(ConfigError, match="Spatial dimensions"):
            StudyBackend(backends=[b1, b2])


class TestStudyBackendData:
    def test_get_all_data(self, two_backends):
        b1, b2, mat1, mat2 = two_backends
        sb = StudyBackend(backends=[b1, b2])
        data = sb.get_data()
        expected = np.vstack([mat1, mat2])
        np.testing.assert_array_almost_equal(data, expected)

    def test_get_data_subset_rows(self, two_backends):
        b1, b2, mat1, mat2 = two_backends
        sb = StudyBackend(backends=[b1, b2])
        # rows from first subject only
        rows = np.array([0, 5, 10], dtype=np.intp)
        data = sb.get_data(rows=rows)
        np.testing.assert_array_almost_equal(data, mat1[rows])

    def test_get_data_cross_subject_rows(self, two_backends):
        b1, b2, mat1, mat2 = two_backends
        sb = StudyBackend(backends=[b1, b2])
        # rows 29 (last of subj 1) and 30 (first of subj 2)
        rows = np.array([29, 30], dtype=np.intp)
        data = sb.get_data(rows=rows)
        np.testing.assert_array_almost_equal(data[0], mat1[29])
        np.testing.assert_array_almost_equal(data[1], mat2[0])

    def test_get_data_subset_cols(self, two_backends):
        b1, b2, mat1, mat2 = two_backends
        sb = StudyBackend(backends=[b1, b2])
        cols = np.array([0, 10, 20], dtype=np.intp)
        data = sb.get_data(cols=cols)
        expected = np.vstack([mat1[:, cols], mat2[:, cols]])
        np.testing.assert_array_almost_equal(data, expected)

    def test_get_data_rows_and_cols(self, two_backends):
        b1, b2, mat1, mat2 = two_backends
        sb = StudyBackend(backends=[b1, b2])
        rows = np.array([0, 35], dtype=np.intp)  # subj1 row 0, subj2 row 5
        cols = np.array([0, 1], dtype=np.intp)
        data = sb.get_data(rows=rows, cols=cols)
        assert data.shape == (2, 2)
        np.testing.assert_array_almost_equal(data[0], mat1[0, cols])
        np.testing.assert_array_almost_equal(data[1], mat2[5, cols])

    def test_get_data_logical_rows(self, two_backends):
        b1, b2, mat1, mat2 = two_backends
        sb = StudyBackend(backends=[b1, b2])
        rows = np.array([True, False, True] + [False] * 47, dtype=bool)
        data = sb.get_data(rows=rows)
        expected = np.vstack([mat1[:], mat2[:]])[[0, 2]]
        np.testing.assert_array_almost_equal(data, expected)

    def test_get_data_rejects_negative_or_oob_rows(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2])
        with pytest.raises(ValueError, match="rows indices must be within"):
            sb.get_data(rows=np.array([-1, 0], dtype=np.intp))
        with pytest.raises(ValueError, match="rows indices must be within"):
            sb.get_data(rows=np.array([50], dtype=np.intp))

    def test_get_data_rejects_invalid_cols(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2])
        with pytest.raises(ValueError, match="cols indices must be within"):
            sb.get_data(cols=np.array([50], dtype=np.intp))
        with pytest.raises(ValueError, match="cols indices must be integers"):
            sb.get_data(cols=np.array([1.5], dtype=float))
        with pytest.raises(ValueError, match="cols indices must be integers"):
            sb.get_data(cols=np.array([1.2], dtype=float))


class TestStudyBackendMask:
    def test_identical_mask(self, rng):
        mask = np.array([True, True, False, True, True], dtype=np.bool_)
        mat1 = rng.standard_normal((10, 5))
        mat2 = rng.standard_normal((10, 5))
        b1 = MatrixBackend(data_matrix=mat1, mask=mask, spatial_dims=(5, 1, 1))
        b2 = MatrixBackend(data_matrix=mat2, mask=mask, spatial_dims=(5, 1, 1))
        sb = StudyBackend(backends=[b1, b2], strict="identical")
        np.testing.assert_array_equal(sb.get_mask(), mask)

    def test_different_mask_identical_raises(self, rng):
        mask1 = np.array([True, True, False, True, True], dtype=np.bool_)
        mask2 = np.array([True, False, False, True, True], dtype=np.bool_)
        mat1 = rng.standard_normal((10, 5))
        mat2 = rng.standard_normal((10, 5))
        b1 = MatrixBackend(data_matrix=mat1, mask=mask1, spatial_dims=(5, 1, 1))
        b2 = MatrixBackend(data_matrix=mat2, mask=mask2, spatial_dims=(5, 1, 1))
        with pytest.raises(ConfigError, match="Mask.*differs"):
            StudyBackend(backends=[b1, b2], strict="identical")

    def test_intersect_mask(self, rng):
        mask1 = np.ones(100, dtype=np.bool_)
        mask2 = np.ones(100, dtype=np.bool_)
        mask2[0] = False  # 99% overlap, above 95% threshold
        mat1 = rng.standard_normal((10, 100))
        mat2 = rng.standard_normal((10, 100))
        b1 = MatrixBackend(data_matrix=mat1, mask=mask1, spatial_dims=(100, 1, 1))
        b2 = MatrixBackend(data_matrix=mat2, mask=mask2, spatial_dims=(100, 1, 1))
        sb = StudyBackend(backends=[b1, b2], strict="intersect")
        expected = mask1 & mask2
        np.testing.assert_array_equal(sb.get_mask(), expected)


class TestStudyBackendLifecycle:
    def test_open_close(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2])
        sb.open()  # should not raise
        sb.close()  # should not raise

    def test_subject_boundaries(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2])
        np.testing.assert_array_equal(
            sb.subject_boundaries, np.array([0, 30, 50], dtype=np.intp)
        )

    def test_metadata(self, two_backends):
        b1, b2, _, _ = two_backends
        sb = StudyBackend(backends=[b1, b2], subject_ids=["A", "B"])
        meta = sb.get_metadata()
        assert meta["storage_format"] == "study"
        assert meta["n_subjects"] == 2
        assert meta["subject_ids"] == ["A", "B"]

    def test_unknown_strict_raises(self, two_backends):
        b1, b2, _, _ = two_backends
        with pytest.raises(ConfigError, match="Unknown strict"):
            StudyBackend(backends=[b1, b2], strict="invalid")
