"""Lazy array conversion via dask.

Port of ``R/as_delayed_array.R``.  Provides :func:`as_dask_array` which
wraps a :class:`StorageBackend` in a ``dask.array.Array`` that lazily
fetches data on demand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend_protocol import StorageBackend
from .errors import ConfigError


def as_dask_array(backend: StorageBackend) -> Any:
    """Convert a storage backend to a lazy ``dask.array.Array``.

    The returned array has shape ``(n_timepoints, n_voxels)`` and fetches
    data from the backend only when a concrete computation is triggered.

    Parameters
    ----------
    backend : StorageBackend
        An opened backend.

    Returns
    -------
    dask.array.Array
        Lazy 2-D array backed by the storage backend.

    Raises
    ------
    ConfigError
        If *dask* is not installed.
    """
    try:
        import dask
        import dask.array as da
    except ImportError as exc:
        raise ConfigError(
            "dask is required for as_dask_array(). "
            "Install with: pip install 'dask[array]'"
        ) from exc

    dims = backend.get_dims()
    n_voxels = int(np.sum(backend.get_mask()))
    n_time = dims.time

    # Choose chunk size: one "run" worth of rows when possible
    if hasattr(backend, "run_lengths"):
        chunk_rows: int = backend.run_lengths[0]
    else:
        chunk_rows = min(n_time, 500)

    # Build blocks using dask.delayed for each row chunk
    blocks: list[Any] = []
    for start in range(0, n_time, chunk_rows):
        end = min(start + chunk_rows, n_time)
        rows_in_chunk = end - start

        _start, _end = start, end

        def _fetch() -> NDArray[np.float64]:
            rows = np.arange(_start, _end, dtype=np.intp)
            data: NDArray[np.float64] = backend.get_data(rows=rows)
            return data

        delayed_fetch: Any = dask.delayed(_fetch)()  # type: ignore[attr-defined]
        block = da.from_delayed(  # type: ignore[attr-defined,no-untyped-call]
            delayed_fetch,
            shape=(rows_in_chunk, n_voxels),
            dtype=np.float64,
        )
        blocks.append(block)

    return da.concatenate(blocks, axis=0)  # type: ignore[attr-defined,no-untyped-call]


def as_dask_array_dataset(dataset: Any) -> Any:
    """Convert an :class:`FmriDataset` to a lazy dask array.

    Convenience wrapper that accesses the dataset's backend.

    Parameters
    ----------
    dataset : FmriDataset
        An active dataset.

    Returns
    -------
    dask.array.Array
    """
    return as_dask_array(dataset._backend)
