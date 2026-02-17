"""Tests for FmriDataset delegating properties added in parity pass.

Covers: samples, run_durations, total_duration, run_indices.
"""

import numpy as np

from fmridataset import matrix_dataset


class TestDelegatingProperties:
    def test_samples(self) -> None:
        mat = np.zeros((20, 5))
        ds = matrix_dataset(mat, TR=2.0, run_length=[10, 10])
        samples = ds.samples
        assert samples.shape == (20,)
        assert samples[0] == 1
        assert samples[-1] == 20

    def test_run_durations(self) -> None:
        mat = np.zeros((30, 5))
        ds = matrix_dataset(mat, TR=1.5, run_length=[10, 20])
        rd = ds.run_durations
        np.testing.assert_allclose(rd, [15.0, 30.0])

    def test_total_duration(self) -> None:
        mat = np.zeros((30, 5))
        ds = matrix_dataset(mat, TR=1.5, run_length=[10, 20])
        assert ds.total_duration == 45.0

    def test_run_indices_run1(self) -> None:
        mat = np.zeros((20, 5))
        ds = matrix_dataset(mat, TR=2.0, run_length=[8, 12])
        idx = ds.run_indices(1)
        np.testing.assert_array_equal(idx, np.arange(8))

    def test_run_indices_run2(self) -> None:
        mat = np.zeros((20, 5))
        ds = matrix_dataset(mat, TR=2.0, run_length=[8, 12])
        idx = ds.run_indices(2)
        np.testing.assert_array_equal(idx, np.arange(8, 20))

    def test_single_run(self) -> None:
        mat = np.zeros((50, 3))
        ds = matrix_dataset(mat, TR=1.0, run_length=50)
        assert ds.total_duration == 50.0
        assert len(ds.run_durations) == 1
        np.testing.assert_array_equal(ds.run_indices(1), np.arange(50))
