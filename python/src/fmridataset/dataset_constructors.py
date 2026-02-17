"""High-level constructors for fMRI datasets.

Port of ``R/dataset_constructors.R``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .backend_registry import BackendRegistry
from .backends.matrix_backend import MatrixBackend
from .dataset import FmriDataset, MatrixDataset
from .sampling_frame import SamplingFrame


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
    datamat = np.asarray(datamat, dtype=np.float64)
    if datamat.ndim == 1:
        datamat = datamat.reshape(-1, 1)
    elif datamat.ndim != 2:
        datamat = np.atleast_2d(datamat)

    if isinstance(run_length, (int, np.integer)):
        run_length = [int(run_length)]
    else:
        run_length = [int(r) for r in run_length]

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
    if isinstance(run_length, (int, np.integer)):
        run_length = [int(run_length)]
    else:
        run_length = [int(r) for r in run_length]

    frame = SamplingFrame.create(blocklens=run_length, TR=TR)

    return FmriDataset(
        backend=backend,
        sampling_frame=frame,
        event_table=event_table,
        censor=censor,
    )
