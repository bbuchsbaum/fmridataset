"""Tests for MatrixBackend."""

import numpy as np
import pytest

from fmridataset import BackendDims, ConfigError, MatrixBackend


class TestMatrixBackendCreation:
    def test_basic(self) -> None:
        mat = np.random.default_rng(0).standard_normal((10, 20))
        mb = MatrixBackend(data_matrix=mat)
        assert mb.get_dims() == BackendDims(spatial=(20, 1, 1), time=10)

    def test_custom_mask(self) -> None:
        mat = np.zeros((5, 10))
        mask = np.array([True] * 6 + [False] * 4)
        mb = MatrixBackend(data_matrix=mat, mask=mask)
        np.testing.assert_array_equal(mb.get_mask(), mask)

    def test_custom_spatial_dims(self) -> None:
        mat = np.zeros((5, 27))
        mb = MatrixBackend(data_matrix=mat, spatial_dims=(3, 3, 3))
        assert mb.get_dims().spatial == (3, 3, 3)

    def test_1d_rejected(self) -> None:
        with pytest.raises(ConfigError, match="2-D"):
            MatrixBackend(data_matrix=np.zeros(10))

    def test_mask_length_mismatch(self) -> None:
        with pytest.raises(ConfigError, match="mask length"):
            MatrixBackend(
                data_matrix=np.zeros((5, 10)),
                mask=np.ones(7, dtype=bool),
            )

    def test_spatial_dims_product_mismatch(self) -> None:
        with pytest.raises(ConfigError, match="prod"):
            MatrixBackend(
                data_matrix=np.zeros((5, 10)),
                spatial_dims=(2, 2, 2),
            )


class TestMatrixBackendData:
    @pytest.fixture()
    def backend(self) -> MatrixBackend:
        rng = np.random.default_rng(1)
        return MatrixBackend(data_matrix=rng.standard_normal((10, 20)))

    def test_get_data_all(self, backend: MatrixBackend) -> None:
        data = backend.get_data()
        assert data.shape == (10, 20)

    def test_get_data_rows(self, backend: MatrixBackend) -> None:
        rows = np.array([0, 2, 4])
        data = backend.get_data(rows=rows)
        assert data.shape == (3, 20)

    def test_get_data_cols(self, backend: MatrixBackend) -> None:
        cols = np.array([0, 5, 10])
        data = backend.get_data(cols=cols)
        assert data.shape == (10, 3)

    def test_get_data_rows_and_cols(self, backend: MatrixBackend) -> None:
        data = backend.get_data(
            rows=np.array([0, 1]),
            cols=np.array([0, 1, 2]),
        )
        assert data.shape == (2, 3)

    def test_mask_applied_on_get_data(self) -> None:
        mat = np.arange(30, dtype=float).reshape(3, 10)
        mask = np.array([True, False] * 5)
        mb = MatrixBackend(data_matrix=mat, mask=mask)
        data = mb.get_data()
        assert data.shape == (3, 5)
        # First masked column should be col 0 of original
        np.testing.assert_array_equal(data[:, 0], mat[:, 0])


class TestMatrixBackendLifecycle:
    def test_context_manager(self) -> None:
        mat = np.zeros((5, 10))
        with MatrixBackend(data_matrix=mat) as mb:
            assert mb.get_data().shape == (5, 10)

    def test_validate(self) -> None:
        mat = np.zeros((5, 10))
        mb = MatrixBackend(data_matrix=mat)
        assert mb.validate() is True

    def test_metadata(self) -> None:
        mat = np.zeros((5, 10))
        mb = MatrixBackend(data_matrix=mat, metadata={"foo": "bar"})
        meta = mb.get_metadata()
        assert meta["foo"] == "bar"
        assert meta["format"] == "matrix"
