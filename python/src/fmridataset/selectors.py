"""Series selectors for spatial sub-setting of fMRI data.

Port of the ``resolve_indices`` generic and its methods from R.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .dataset import FmriDataset


class SeriesSelector(ABC):
    """Base class for spatial selectors."""

    @abstractmethod
    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        """Resolve to 0-based column indices within the masked voxel array."""


class IndexSelector(SeriesSelector):
    """Select voxels by 0-based column indices within the masked data.

    Parameters
    ----------
    indices : array-like of int
        0-based column indices.
    """

    def __init__(self, indices: NDArray[np.intp] | list[int]) -> None:
        self.indices = np.asarray(indices, dtype=np.intp)

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        n_voxels = int(dataset.get_mask().sum())
        if np.any(self.indices < 0) or np.any(self.indices >= n_voxels):
            raise IndexError(
                f"Index out of range [0, {n_voxels})"
            )
        return self.indices


class AllSelector(SeriesSelector):
    """Select all voxels within the mask."""

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        n_voxels = int(dataset.get_mask().sum())
        return np.arange(n_voxels, dtype=np.intp)


class ROISelector(SeriesSelector):
    """Select voxels falling within an ROI mask.

    Parameters
    ----------
    roi_mask : ndarray of bool
        Boolean mask with the same spatial shape as the dataset mask.
    """

    def __init__(self, roi_mask: NDArray[np.bool_]) -> None:
        self.roi_mask = np.asarray(roi_mask, dtype=np.bool_).ravel()

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dataset_mask = dataset.get_mask()

        if self.roi_mask.size != dataset_mask.size:
            raise ValueError(
                f"ROI mask length ({self.roi_mask.size}) must equal "
                f"dataset mask length ({dataset_mask.size})"
            )

        # Indices of voxels that are in BOTH the dataset mask and the ROI
        dataset_voxels = np.where(dataset_mask)[0]
        roi_voxels = set(np.where(self.roi_mask)[0].tolist())

        cols: list[int] = []
        for col_idx, vox_idx in enumerate(dataset_voxels):
            if vox_idx in roi_voxels:
                cols.append(col_idx)

        return np.array(cols, dtype=np.intp)
