"""Tests for data_chunks and ChunkIterator."""

import numpy as np
import pytest

from fmridataset import (
    DataChunk,
    collect_chunks,
    data_chunks,
    matrix_dataset,
)


@pytest.fixture()
def dataset():
    rng = np.random.default_rng(7)
    mat = rng.standard_normal((100, 50))
    return matrix_dataset(mat, TR=2.0, run_length=[50, 50])


class TestSingleChunk:
    def test_one_chunk(self, dataset) -> None:
        it = data_chunks(dataset, nchunks=1)
        assert len(it) == 1
        chunks = it.collect()
        assert len(chunks) == 1
        assert chunks[0].data.shape == (100, 50)
        assert chunks[0].chunk_num == 1

    def test_voxel_and_row_indices(self, dataset) -> None:
        chunk = next(iter(data_chunks(dataset, nchunks=1)))
        np.testing.assert_array_equal(chunk.voxel_ind, np.arange(50))
        np.testing.assert_array_equal(chunk.row_ind, np.arange(100))


class TestMultipleChunks:
    def test_four_chunks(self, dataset) -> None:
        it = data_chunks(dataset, nchunks=4)
        assert len(it) == 4
        chunks = it.collect()
        # All voxels should be covered
        all_voxels = np.concatenate([c.voxel_ind for c in chunks])
        assert len(np.unique(all_voxels)) == 50
        # Each chunk should have all timepoints
        for c in chunks:
            assert c.data.shape[0] == 100

    def test_more_chunks_than_voxels(self) -> None:
        mat = np.zeros((10, 3))
        ds = matrix_dataset(mat, TR=1.0, run_length=10)
        with pytest.warns(UserWarning, match="3 chunks"):
            it = data_chunks(ds, nchunks=100)
        assert len(it) == 3

    def test_chunk_data_reconstruction(self, dataset) -> None:
        """Reconstructed matrix from chunks should match original."""
        chunks = data_chunks(dataset, nchunks=5).collect()
        reconstructed = np.zeros_like(dataset.datamat)
        for c in chunks:
            reconstructed[:, c.voxel_ind] = c.data
        np.testing.assert_array_almost_equal(reconstructed, dataset.datamat)


class TestRunwiseChunks:
    def test_runwise(self, dataset) -> None:
        it = data_chunks(dataset, runwise=True)
        assert len(it) == 2
        chunks = it.collect()
        assert chunks[0].data.shape[0] == 50
        assert chunks[1].data.shape[0] == 50

    def test_runwise_row_indices(self, dataset) -> None:
        chunks = data_chunks(dataset, runwise=True).collect()
        np.testing.assert_array_equal(chunks[0].row_ind, np.arange(50))
        np.testing.assert_array_equal(chunks[1].row_ind, np.arange(50, 100))

    def test_three_runs(self) -> None:
        mat = np.zeros((30, 10))
        ds = matrix_dataset(mat, TR=1.0, run_length=[10, 10, 10])
        chunks = data_chunks(ds, runwise=True).collect()
        assert len(chunks) == 3
        for i, c in enumerate(chunks):
            assert c.data.shape[0] == 10
            assert c.chunk_num == i + 1


class TestCollectChunks:
    def test_collect_function(self, dataset) -> None:
        it = data_chunks(dataset, nchunks=3)
        chunks = collect_chunks(it)
        assert len(chunks) == 3


class TestChunkIteratorProtocol:
    def test_iteration(self, dataset) -> None:
        it = data_chunks(dataset, nchunks=2)
        count = 0
        for chunk in it:
            count += 1
            assert isinstance(chunk, DataChunk)
        assert count == 2

    def test_len(self, dataset) -> None:
        it = data_chunks(dataset, nchunks=4)
        assert len(it) == 4
