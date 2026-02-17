"""FmriSeries: lazy time-series container.

Port of ``R/FmriSeries.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class FmriSeries:
    """Container for lazily-accessed fMRI time-series data.

    Parameters
    ----------
    data : ndarray
        2-D array shaped ``(n_timepoints, n_voxels)``.
    voxel_info : DataFrame
        One row per voxel with spatial metadata.
    temporal_info : DataFrame
        One row per time-point with temporal metadata.
    selection_info : dict
        Description of how the data were selected.
    dataset_info : dict
        Description of the source dataset / backend.
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        voxel_info: pd.DataFrame,
        temporal_info: pd.DataFrame,
        selection_info: dict[str, Any] | None = None,
        dataset_info: dict[str, Any] | None = None,
    ) -> None:
        self._data = np.asarray(data)
        self._voxel_info = voxel_info
        self._temporal_info = temporal_info
        self._selection_info = selection_info or {}
        self._dataset_info = dataset_info or {}

        if self._data.ndim != 2:
            raise ValueError("data must be a 2-D array")
        if len(voxel_info) != self._data.shape[1]:
            raise ValueError(
                f"voxel_info rows ({len(voxel_info)}) must equal "
                f"data columns ({self._data.shape[1]})"
            )
        if len(temporal_info) != self._data.shape[0]:
            raise ValueError(
                f"temporal_info rows ({len(temporal_info)}) must equal "
                f"data rows ({self._data.shape[0]})"
            )

    # -- properties --------------------------------------------------------

    @property
    def data(self) -> NDArray[np.floating[Any]]:
        return self._data

    @property
    def voxel_info(self) -> pd.DataFrame:
        return self._voxel_info

    @property
    def temporal_info(self) -> pd.DataFrame:
        return self._temporal_info

    @property
    def selection_info(self) -> dict[str, Any]:
        return self._selection_info

    @property
    def dataset_info(self) -> dict[str, Any]:
        return self._dataset_info

    @property
    def shape(self) -> tuple[int, int]:
        s = self._data.shape
        return (s[0], s[1])

    # -- conversions -------------------------------------------------------

    def to_numpy(self) -> NDArray[np.floating[Any]]:
        """Return the data as a plain ndarray."""
        return np.array(self._data)

    def to_dataframe(self) -> pd.DataFrame:
        """Return long-form DataFrame (one row per voxel * timepoint).

        Columns from ``temporal_info`` and ``voxel_info`` are repeated
        as appropriate, plus a ``signal`` column with the data values.
        """
        n_time, n_vox = self._data.shape
        time_idx = np.repeat(np.arange(n_time), n_vox)
        vox_idx = np.tile(np.arange(n_vox), n_time)

        out = pd.concat(
            [
                self._temporal_info.iloc[time_idx].reset_index(drop=True),
                self._voxel_info.iloc[vox_idx].reset_index(drop=True),
            ],
            axis=1,
        )
        out["signal"] = self._data.ravel()
        return out

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        n_vox, n_time = self._data.shape[1], self._data.shape[0]
        sel = "custom" if self._selection_info else "all"
        backend = self._dataset_info.get("backend_type", "?")
        return (
            f"<FmriSeries {n_vox} voxels x {n_time} timepoints | "
            f"selector={sel} backend={backend}>"
        )

    def __len__(self) -> int:
        return int(self._data.shape[0])
