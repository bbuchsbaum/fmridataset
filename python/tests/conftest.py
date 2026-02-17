"""Shared fixtures for fmridataset tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmridataset import (
    MatrixBackend,
    MatrixDataset,
    SamplingFrame,
    matrix_dataset,
)


@pytest.fixture()
def zarr_store(tmp_path, rng):
    """Create a small 4D zarr array on disk."""
    zarr = pytest.importorskip("zarr")
    nx, ny, nz, nt = 3, 3, 3, 10
    data_4d = rng.standard_normal((nx, ny, nz, nt)).astype(np.float64)
    store_path = str(tmp_path / "test.zarr")
    zarr_array = zarr.open(store_path, mode="w", shape=(nx, ny, nz, nt), dtype=np.float64)
    zarr_array[:] = data_4d
    return store_path


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def simple_matrix(rng: np.random.Generator) -> np.ndarray:
    """100 timepoints x 50 voxels."""
    return rng.standard_normal((100, 50))


@pytest.fixture()
def simple_frame() -> SamplingFrame:
    return SamplingFrame.create(blocklens=[50, 50], TR=2.0)


@pytest.fixture()
def simple_backend(simple_matrix: np.ndarray) -> MatrixBackend:
    return MatrixBackend(data_matrix=simple_matrix)


@pytest.fixture()
def simple_dataset(simple_matrix: np.ndarray) -> MatrixDataset:
    return matrix_dataset(
        datamat=simple_matrix,
        TR=2.0,
        run_length=[50, 50],
    )


# -- rpy2 fixtures (session-scoped, skipped if R not available) -----------

@pytest.fixture(scope="session")
def r_fmridataset():
    """Load R fmridataset package via rpy2."""
    pytest.importorskip("rpy2")
    from rpy2.robjects.packages import importr

    return importr("fmridataset")


@pytest.fixture(scope="session")
def r_fmrihrf():
    """Load R fmrihrf package via rpy2."""
    pytest.importorskip("rpy2")
    from rpy2.robjects.packages import importr

    return importr("fmrihrf")
