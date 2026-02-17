"""High-level constructors for fMRI datasets.

Port of ``R/dataset_constructors.R``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pathlib import Path

from .backend_registry import BackendRegistry
from .backends.matrix_backend import MatrixBackend
from .dataset import FmriDataset, MatrixDataset
from .sampling_frame import SamplingFrame


def _coerce_run_length(run_length: int | Sequence[int]) -> list[int]:
    """Normalize and validate run lengths to positive integer values."""

    values = [run_length] if isinstance(run_length, (int, np.integer)) else list(run_length)
    if len(values) == 0:
        raise ValueError("run_length must be a non-empty sequence of positive integers")

    normalized: list[int] = []
    for value in values:
        if isinstance(value, (str, bytes)):
            raise ValueError("run_length values must be numeric")

        try:
            value_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("run_length values must be numeric") from exc

        if not value_float.is_integer():
            raise ValueError("run_length values must be integers")

        value_int = int(value_float)
        if value_int <= 0:
            raise ValueError("all run_length values must be positive")

        normalized.append(value_int)

    return normalized


def matrix_dataset(
    datamat: NDArray[np.floating[Any]],
    TR: float,  # noqa: N803
    run_length: int | Sequence[int],
    event_table: pd.DataFrame | None = None,
) -> MatrixDataset:
    """Create a dataset from an in-memory matrix.

    Parameters
    ----------
    datamat : ndarray
        2-D array shaped ``(n_timepoints, n_voxels)``.
    TR : float
        Repetition time in seconds.
    run_length : int or sequence of int
        Number of time-points per run.
    event_table : DataFrame or None
        Optional event information.

    Returns
    -------
    MatrixDataset
    """
    if event_table is not None:
        event_table = _ensure_event_table_unique(event_table)

    datamat = np.asarray(datamat, dtype=np.float64)
    if datamat.ndim == 1:
        datamat = datamat.reshape(-1, 1)
    elif datamat.ndim != 2:
        raise ValueError("datamat must be a 2-D matrix or 1-D vector")

    run_length = _coerce_run_length(run_length)

    if sum(run_length) != datamat.shape[0]:
        raise ValueError(
            f"sum(run_length) ({sum(run_length)}) must equal "
            f"number of rows ({datamat.shape[0]})"
        )

    frame = SamplingFrame.create(blocklens=run_length, TR=TR)
    mask = np.ones(datamat.shape[1], dtype=np.bool_)
    backend = MatrixBackend(data_matrix=datamat, mask=mask)

    return MatrixDataset(
        backend=backend,
        sampling_frame=frame,
        datamat=datamat,
        event_table=event_table,
    )


def _ensure_event_table_unique(event_table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ensure event table columns are uniquely named."""

    if event_table is None:
        return None

    duplicates = pd.Index(event_table.columns)[pd.Index(event_table.columns).duplicated()]
    if len(duplicates):
        duplicate_list = ", ".join(sorted(set(map(str, duplicates))))
        raise ValueError(
            f"event_table columns must be unique, duplicate columns: {duplicate_list}"
        )

    return event_table


def fmri_dataset(
    backend: Any,
    TR: float,  # noqa: N803
    run_length: int | Sequence[int],
    event_table: pd.DataFrame | None = None,
    censor: NDArray[np.intp] | None = None,
) -> FmriDataset:
    """Create an FmriDataset from an already-constructed backend.

    Parameters
    ----------
    backend : StorageBackend
        An opened storage backend instance.
    TR : float
        Repetition time in seconds.
    run_length : int or sequence of int
        Number of time-points per run.
    event_table : DataFrame or None
        Optional event information.
    censor : ndarray or None
        Binary censor vector.

    Returns
    -------
    FmriDataset
    """
    run_length = _coerce_run_length(run_length)

    frame = SamplingFrame.create(blocklens=run_length, TR=TR)

    return FmriDataset(
        backend=backend,
        sampling_frame=frame,
        event_table=_ensure_event_table_unique(event_table),
        censor=censor,
    )


def zarr_dataset(
    zarr_source: str | Path,
    TR: float,  # noqa: N803
    run_length: int | Sequence[int],
    event_table: pd.DataFrame | None = None,
    censor: NDArray[np.intp] | None = None,
    preload: bool = False,
) -> FmriDataset:
    """Create an FmriDataset from a Zarr array store.

    Parameters
    ----------
    zarr_source : str or Path
        Path to Zarr store (directory or URL).
    TR : float
        Repetition time in seconds.
    run_length : int or sequence of int
        Number of time-points per run.
    event_table : DataFrame or None
        Optional event information.
    censor : ndarray or None
        Binary censor vector.
    preload : bool
        If True, eagerly load all data into memory.

    Returns
    -------
    FmriDataset
    """
    from .backends.zarr_backend import ZarrBackend

    backend = ZarrBackend(source=str(zarr_source), preload=preload)
    backend.open()

    return fmri_dataset(
        backend=backend,
        TR=TR,
        run_length=run_length,
        event_table=event_table,
        censor=censor,
    )


def fmri_zarr_dataset(
    zarr_source: str | Path,
    TR: float,  # noqa: N803
    run_length: int | Sequence[int],
    event_table: pd.DataFrame | None = None,
    censor: NDArray[np.intp] | None = None,
    preload: bool = False,
) -> FmriDataset:
    """Create an FmriDataset from a Zarr array store.

    Alias of :func:`zarr_dataset` to preserve R API parity.
    """

    return zarr_dataset(
        zarr_source=zarr_source,
        TR=TR,
        run_length=run_length,
        event_table=event_table,
        censor=censor,
        preload=preload,
    )
