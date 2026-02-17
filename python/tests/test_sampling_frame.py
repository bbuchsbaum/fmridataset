"""Tests for SamplingFrame."""

import numpy as np
import pytest

from fmridataset import SamplingFrame


class TestSamplingFrameBasic:
    def test_create(self) -> None:
        sf = SamplingFrame.create(blocklens=[100, 120], TR=2.0)
        assert sf.blocklens == (100, 120)
        assert sf.TR == 2.0

    def test_n_runs(self) -> None:
        sf = SamplingFrame.create(blocklens=[100, 120, 110], TR=2.0)
        assert sf.n_runs == 3

    def test_n_timepoints(self) -> None:
        sf = SamplingFrame.create(blocklens=[100, 120], TR=2.0)
        assert sf.n_timepoints == 220

    def test_blockids(self) -> None:
        sf = SamplingFrame.create(blocklens=[3, 4], TR=2.0)
        expected = np.array([1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(sf.blockids, expected)

    def test_samples(self) -> None:
        sf = SamplingFrame.create(blocklens=[3, 4], TR=2.0)
        expected = np.arange(1, 8)
        np.testing.assert_array_equal(sf.samples, expected)

    def test_run_durations(self) -> None:
        sf = SamplingFrame.create(blocklens=[100, 120], TR=2.0)
        np.testing.assert_array_equal(sf.run_durations, [200.0, 240.0])

    def test_total_duration(self) -> None:
        sf = SamplingFrame.create(blocklens=[100, 120], TR=2.0)
        assert sf.total_duration == 440.0

    def test_run_indices(self) -> None:
        sf = SamplingFrame.create(blocklens=[3, 4], TR=2.0)
        np.testing.assert_array_equal(sf.run_indices(1), [0, 1, 2])
        np.testing.assert_array_equal(sf.run_indices(2), [3, 4, 5, 6])

    def test_frozen(self) -> None:
        sf = SamplingFrame.create(blocklens=[100], TR=2.0)
        with pytest.raises(AttributeError):
            sf.TR = 3.0  # type: ignore[misc]

    def test_single_run(self) -> None:
        sf = SamplingFrame.create(blocklens=[200], TR=1.5)
        assert sf.n_runs == 1
        assert sf.n_timepoints == 200
        assert sf.total_duration == 300.0


class TestSamplingFrameValidation:
    def test_empty_blocklens(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SamplingFrame.create(blocklens=[], TR=2.0)

    def test_negative_blocklen(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SamplingFrame.create(blocklens=[100, -1], TR=2.0)

    def test_zero_blocklen(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SamplingFrame.create(blocklens=[0], TR=2.0)

    def test_negative_TR(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SamplingFrame.create(blocklens=[100], TR=-1.0)

    def test_zero_TR(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SamplingFrame.create(blocklens=[100], TR=0.0)

    def test_run_indices_out_of_range(self) -> None:
        sf = SamplingFrame.create(blocklens=[3, 4], TR=2.0)
        with pytest.raises(ValueError):
            sf.run_indices(0)
        with pytest.raises(ValueError):
            sf.run_indices(3)


class TestSamplingFrameRepr:
    def test_repr(self) -> None:
        sf = SamplingFrame.create(blocklens=[100, 120], TR=2.0)
        r = repr(sf)
        assert "n_runs=2" in r
        assert "n_timepoints=220" in r
        assert "TR=2.0" in r
