"""Core FmriDataset classes.

Port of the dataset structure from ``R/dataset_constructors.R`` and
``R/data_access.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .backend_protocol import StorageBackend
from .errors import ConfigError
from .sampling_frame import SamplingFrame


class FmriDataset:
    """Unified fMRI dataset container.

    Wraps a :class:`StorageBackend` together with a :class:`SamplingFrame`
    and optional event / censor information.

    Parameters
    ----------
    backend : StorageBackend
        Opened storage backend.
    sampling_frame : SamplingFrame
        Temporal structure.
    event_table : DataFrame or None
        Stimulus / event information.
    censor : ndarray of int or None
        Binary vector (0/1) marking time-points to censor.
    """

    def __init__(
        self,
        backend: StorageBackend,
        sampling_frame: SamplingFrame,
        event_table: pd.DataFrame | None = None,
        censor: NDArray[np.intp] | None = None,
    ) -> None:
        self._backend = backend
        self._sampling_frame = sampling_frame
        if event_table is None:
            event_table = pd.DataFrame()
        self._event_table = event_table

        if censor is None:
            self._censor = np.zeros(sampling_frame.n_timepoints, dtype=np.intp)
        else:
            self._censor = np.asarray(censor, dtype=np.intp)

        # Validate time dimension matches
        dims = backend.get_dims()
        if sampling_frame.n_timepoints != dims.time:
            raise ConfigError(
                f"sampling_frame n_timepoints ({sampling_frame.n_timepoints}) "
                f"!= backend time dimension ({dims.time})"
            )

    # -- delegating properties ---------------------------------------------

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    @property
    def sampling_frame(self) -> SamplingFrame:
        return self._sampling_frame

    @property
    def event_table(self) -> pd.DataFrame:
        return self._event_table

    @property
    def censor(self) -> NDArray[np.intp]:
        return self._censor

    @property
    def TR(self) -> float:  # noqa: N802
        return self._sampling_frame.TR

    @property
    def n_runs(self) -> int:
        return self._sampling_frame.n_runs

    @property
    def n_timepoints(self) -> int:
        return self._sampling_frame.n_timepoints

    @property
    def blocklens(self) -> tuple[int, ...]:
        return self._sampling_frame.blocklens

    @property
    def blockids(self) -> NDArray[np.intp]:
        return self._sampling_frame.blockids

    # -- data access -------------------------------------------------------

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Read data from the backend (timepoints x voxels)."""
        return self._backend.get_data(rows=rows, cols=cols)

    def get_data_matrix(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Alias of :meth:`get_data` – always returns a 2-D ndarray."""
        return self.get_data(rows=rows, cols=cols)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the backend's boolean mask."""
        return self._backend.get_mask()

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        dims = self._backend.get_dims()
        n_vox = int(self.get_mask().sum())
        return (
            f"<FmriDataset "
            f"spatial={dims.spatial} time={dims.time} "
            f"voxels={n_vox} runs={self.n_runs} TR={self.TR}>"
        )


class MatrixDataset(FmriDataset):
    """Dataset backed by an in-memory matrix.

    Convenience subclass that stores the raw data matrix as ``datamat``
    for backward-compatible direct access.
    """

    def __init__(
        self,
        backend: StorageBackend,
        sampling_frame: SamplingFrame,
        datamat: NDArray[np.floating[Any]],
        event_table: pd.DataFrame | None = None,
        censor: NDArray[np.intp] | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            sampling_frame=sampling_frame,
            event_table=event_table,
            censor=censor,
        )
        self._datamat = datamat

    @property
    def datamat(self) -> NDArray[np.floating[Any]]:
        """Direct access to the underlying data matrix."""
        return self._datamat

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        mat = self._datamat
        if rows is not None:
            mat = mat[rows, :]
        if cols is not None:
            mat = mat[:, cols]
        return mat

    def __repr__(self) -> str:
        return (
            f"<MatrixDataset "
            f"shape={self._datamat.shape} "
            f"runs={self.n_runs} TR={self.TR}>"
        )
