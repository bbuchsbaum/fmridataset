"""Tests for FmriSeries."""

import numpy as np
import pandas as pd
import pytest

from fmridataset import FmriSeries


@pytest.fixture()
def series() -> FmriSeries:
    data = np.random.default_rng(5).standard_normal((10, 4))
    voxel_info = pd.DataFrame({"voxel_id": range(4)})
    temporal_info = pd.DataFrame({"time": range(10)})
    return FmriSeries(
        data=data,
        voxel_info=voxel_info,
        temporal_info=temporal_info,
        selection_info={"selector": "test"},
        dataset_info={"backend_type": "matrix"},
    )


class TestFmriSeriesBasic:
    def test_shape(self, series: FmriSeries) -> None:
        assert series.shape == (10, 4)

    def test_len(self, series: FmriSeries) -> None:
        assert len(series) == 10

    def test_to_numpy(self, series: FmriSeries) -> None:
        arr = series.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 4)

    def test_to_dataframe(self, series: FmriSeries) -> None:
        df = series.to_dataframe()
        assert len(df) == 40  # 10 * 4
        assert "signal" in df.columns
        assert "time" in df.columns
        assert "voxel_id" in df.columns


class TestFmriSeriesValidation:
    def test_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            FmriSeries(
                data=np.zeros(10),
                voxel_info=pd.DataFrame(),
                temporal_info=pd.DataFrame(),
            )

    def test_voxel_info_mismatch(self) -> None:
        with pytest.raises(ValueError, match="voxel_info"):
            FmriSeries(
                data=np.zeros((10, 4)),
                voxel_info=pd.DataFrame({"x": range(3)}),
                temporal_info=pd.DataFrame({"t": range(10)}),
            )

    def test_temporal_info_mismatch(self) -> None:
        with pytest.raises(ValueError, match="temporal_info"):
            FmriSeries(
                data=np.zeros((10, 4)),
                voxel_info=pd.DataFrame({"x": range(4)}),
                temporal_info=pd.DataFrame({"t": range(5)}),
            )


class TestFmriSeriesRepr:
    def test_repr(self, series: FmriSeries) -> None:
        r = repr(series)
        assert "FmriSeries" in r
        assert "4 voxels" in r
        assert "10 timepoints" in r
