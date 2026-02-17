"""Chunked iteration over fMRI datasets.

Port of ``R/data_chunks.R``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from .dataset import FmriDataset, MatrixDataset


@dataclass
class DataChunk:
    """A single chunk of data extracted from a dataset.

    Attributes
    ----------
    data : ndarray
        2-D array shaped ``(n_rows, n_cols)``.
    voxel_ind : ndarray
        0-based voxel indices included in this chunk.
    row_ind : ndarray
        0-based row (time-point) indices included in this chunk.
    chunk_num : int
        1-based chunk number.
    """

    data: NDArray[np.floating[Any]]
    voxel_ind: NDArray[np.intp]
    row_ind: NDArray[np.intp]
    chunk_num: int


class ChunkIterator:
    """Iterable that yields :class:`DataChunk` objects.

    Supports ``len()``, iteration, and a :meth:`collect` helper.
    """

    def __init__(
        self,
        nchunks: int,
        get_chunk: Callable[[int], DataChunk],
    ) -> None:
        self._nchunks = nchunks
        self._get_chunk = get_chunk
        self._current = 0

    @property
    def nchunks(self) -> int:
        return self._nchunks

    def __len__(self) -> int:
        return self._nchunks

    def __iter__(self) -> Iterator[DataChunk]:
        self._current = 0
        return self

    def __next__(self) -> DataChunk:
        if self._current >= self._nchunks:
            raise StopIteration
        # Chunks are 1-based in the public API
        chunk = self._get_chunk(self._current + 1)
        self._current += 1
        return chunk

    def collect(self) -> list[DataChunk]:
        """Materialise all chunks into a list."""
        return list(self)


# -----------------------------------------------------------------------
# Strategy functions
# -----------------------------------------------------------------------


def data_chunks(
    dataset: FmriDataset,
    nchunks: int = 1,
    runwise: bool = False,
) -> ChunkIterator:
    """Create a chunk iterator for *dataset*.

    Parameters
    ----------
    dataset : FmriDataset
        The dataset to chunk.
    nchunks : int
        Number of equal-sized voxel chunks (ignored when *runwise=True*).
    runwise : bool
        If True, create one chunk per run.
    """
    if isinstance(dataset, MatrixDataset):
        return _matrix_chunks(dataset, nchunks=nchunks, runwise=runwise)
    return _generic_chunks(dataset, nchunks=nchunks, runwise=runwise)


def _matrix_chunks(
    ds: MatrixDataset,
    nchunks: int,
    runwise: bool,
) -> ChunkIterator:
    """Chunk strategy for :class:`MatrixDataset`."""
    mat = ds.datamat
    sf = ds.sampling_frame

    if runwise:
        def get_run_chunk(chunk_num: int) -> DataChunk:
            # chunk_num is 1-based
            row_ind = sf.run_indices(chunk_num)
            return DataChunk(
                data=mat[row_ind, :],
                voxel_ind=np.arange(mat.shape[1], dtype=np.intp),
                row_ind=row_ind,
                chunk_num=chunk_num,
            )

        return ChunkIterator(sf.n_runs, get_run_chunk)

    if nchunks == 1:
        def get_one(chunk_num: int) -> DataChunk:
            return DataChunk(
                data=mat,
                voxel_ind=np.arange(mat.shape[1], dtype=np.intp),
                row_ind=np.arange(mat.shape[0], dtype=np.intp),
                chunk_num=chunk_num,
            )

        return ChunkIterator(1, get_one)

    # Arbitrary voxel-wise chunks
    n_vox = mat.shape[1]
    if nchunks > n_vox:
        warnings.warn(
            f"requested {nchunks} chunks but only {n_vox} voxels; "
            f"using {n_vox} chunks instead",
            stacklevel=2,
        )
        nchunks = n_vox

    splits = _split_indices(n_vox, nchunks)

    def get_arb_chunk(chunk_num: int) -> DataChunk:
        col_idx = splits[chunk_num - 1]
        return DataChunk(
            data=mat[:, col_idx],
            voxel_ind=col_idx,
            row_ind=np.arange(mat.shape[0], dtype=np.intp),
            chunk_num=chunk_num,
        )

    return ChunkIterator(nchunks, get_arb_chunk)


def _generic_chunks(
    ds: FmriDataset,
    nchunks: int,
    runwise: bool,
) -> ChunkIterator:
    """Chunk strategy for generic :class:`FmriDataset` (backend-based)."""
    backend = ds.backend
    sf = ds.sampling_frame
    mask = backend.get_mask()
    voxel_ind = np.where(mask)[0]
    n_voxels = int(mask.sum())
    dims = backend.get_dims()

    if runwise:
        def get_run_chunk(chunk_num: int) -> DataChunk:
            row_ind = sf.run_indices(chunk_num)
            mat = backend.get_data(rows=row_ind, cols=None)
            return DataChunk(
                data=mat,
                voxel_ind=voxel_ind,
                row_ind=row_ind,
                chunk_num=chunk_num,
            )

        return ChunkIterator(sf.n_runs, get_run_chunk)

    if nchunks == 1:
        def get_one(chunk_num: int) -> DataChunk:
            mat = backend.get_data()
            return DataChunk(
                data=mat,
                voxel_ind=voxel_ind,
                row_ind=np.arange(dims.time, dtype=np.intp),
                chunk_num=chunk_num,
            )

        return ChunkIterator(1, get_one)

    if nchunks > n_voxels:
        warnings.warn(
            f"requested {nchunks} chunks but only {n_voxels} voxels; "
            f"using {n_voxels} chunks instead",
            stacklevel=2,
        )
        nchunks = n_voxels

    splits = _split_indices(n_voxels, nchunks)

    def get_arb_chunk(chunk_num: int) -> DataChunk:
        col_idx = splits[chunk_num - 1]
        mat = backend.get_data(rows=None, cols=col_idx)
        return DataChunk(
            data=mat,
            voxel_ind=voxel_ind[col_idx] if col_idx.max() < len(voxel_ind) else col_idx,
            row_ind=np.arange(dims.time, dtype=np.intp),
            chunk_num=chunk_num,
        )

    return ChunkIterator(nchunks, get_arb_chunk)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _split_indices(n: int, nchunks: int) -> list[NDArray[np.intp]]:
    """Split ``range(n)`` into *nchunks* roughly-equal groups.

    Matches R's ``sort(rep(1:nchunks, length.out=n))`` splitting.
    """
    assignments = np.sort(
        np.tile(np.arange(nchunks), (n + nchunks - 1) // nchunks)[:n]
    )
    return [
        np.where(assignments == i)[0].astype(np.intp)
        for i in range(nchunks)
    ]


def collect_chunks(iterator: ChunkIterator) -> list[DataChunk]:
    """Materialise all chunks from *iterator* into a list."""
    return iterator.collect()
