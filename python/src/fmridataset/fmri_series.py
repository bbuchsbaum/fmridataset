"""FmriSeries: lazy time-series container and query function.

Port of ``R/FmriSeries.R``, ``R/fmri_series.R``,
``R/fmri_series_resolvers.R``, and ``R/fmri_series_metadata.R``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .dataset import FmriDataset
    from .selectors import SeriesSelector


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


# ---------------------------------------------------------------------------
# Resolver helpers (port of R/fmri_series_resolvers.R)
# ---------------------------------------------------------------------------


def resolve_selector(
    dataset: FmriDataset,
    selector: SeriesSelector | NDArray[np.intp] | None,
) -> NDArray[np.intp]:
    """Resolve a spatial selector to 0-based column indices.

    Parameters
    ----------
    dataset : FmriDataset
        The dataset to resolve against.
    selector : SeriesSelector, ndarray, or None
        ``None`` selects all voxels. An integer array is used directly.
        A :class:`SeriesSelector` is resolved via its ``resolve_indices``
        method.
    """
    from .selectors import SeriesSelector as _SS

    if selector is None:
        n_voxels = int(dataset.get_mask().sum())
        return np.arange(n_voxels, dtype=np.intp)

    if isinstance(selector, _SS):
        return selector.resolve_indices(dataset)

    arr = np.asarray(selector)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.intp, copy=False)

    raise ValueError(f"Unsupported selector type: {type(selector)}")


def resolve_timepoints(
    dataset: FmriDataset,
    timepoints: NDArray[np.intp] | NDArray[np.bool_] | None,
) -> NDArray[np.intp]:
    """Resolve a temporal selection to 0-based row indices.

    Parameters
    ----------
    dataset : FmriDataset
        The dataset to resolve against.
    timepoints : ndarray or None
        ``None`` selects all timepoints. A boolean array is converted via
        ``np.where``. An integer array is used directly (0-based).
    """
    n_time = dataset.n_timepoints
    if timepoints is None:
        return np.arange(n_time, dtype=np.intp)

    arr = np.asarray(timepoints)
    if np.issubdtype(arr.dtype, np.bool_):
        if arr.size != n_time:
            raise ValueError(
                f"Boolean timepoints length ({arr.size}) must equal "
                f"n_timepoints ({n_time})"
            )
        return np.where(arr)[0].astype(np.intp)

    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.intp, copy=False)

    raise ValueError(f"Unsupported timepoints type: {arr.dtype}")


# ---------------------------------------------------------------------------
# Temporal metadata builder (port of R/fmri_series_metadata.R)
# ---------------------------------------------------------------------------


def _build_temporal_info(
    dataset: FmriDataset,
    time_indices: NDArray[np.intp],
) -> pd.DataFrame:
    """Build a ``temporal_info`` DataFrame for selected timepoints."""
    blockids = dataset.blockids
    return pd.DataFrame(
        {
            "run_id": blockids[time_indices],
            "timepoint": time_indices,
        }
    )


# ---------------------------------------------------------------------------
# Top-level query function (port of R/fmri_series.R)
# ---------------------------------------------------------------------------


def fmri_series(
    dataset: FmriDataset,
    selector: SeriesSelector | NDArray[np.intp] | None = None,
    timepoints: NDArray[np.intp] | NDArray[np.bool_] | None = None,
) -> FmriSeries:
    """Query fMRI time-series from a dataset.

    Parameters
    ----------
    dataset : FmriDataset
        Source dataset.
    selector : SeriesSelector, ndarray, or None
        Spatial selection. ``None`` selects all voxels.
    timepoints : ndarray or None
        Temporal selection. ``None`` selects all timepoints.

    Returns
    -------
    FmriSeries
        Container with data, voxel_info, and temporal_info.
    """
    voxel_ind = resolve_selector(dataset, selector)
    time_ind = resolve_timepoints(dataset, timepoints)

    data = dataset.get_data(rows=time_ind, cols=voxel_ind)

    voxel_info = pd.DataFrame({"voxel": voxel_ind})
    temporal_info = _build_temporal_info(dataset, time_ind)

    backend_type = type(dataset._backend).__name__

    return FmriSeries(
        data=data,
        voxel_info=voxel_info,
        temporal_info=temporal_info,
        selection_info={
            "selector": repr(selector) if selector is not None else None,
            "timepoints": repr(timepoints) if timepoints is not None else None,
        },
        dataset_info={"backend_type": backend_type},
    )
