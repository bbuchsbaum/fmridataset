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


class VoxelSelector(SeriesSelector):
    """Select voxels by 3-D coordinates in volume space.

    Parameters
    ----------
    coords : ndarray, shape (N, 3) or (3,)
        Voxel coordinates (x, y, z). 1-based to match R convention.
    """

    def __init__(self, coords: NDArray[np.intp] | list[list[int]]) -> None:
        arr = np.asarray(coords, dtype=np.intp)
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError("Single coordinate must have length 3")
            arr = arr.reshape(1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        self.coords = arr

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dims = dataset.get_dims().spatial
        mask = dataset.get_mask()

        # Validate coordinates (1-based)
        for axis, dim_size in enumerate(dims):
            if np.any(self.coords[:, axis] < 1) or np.any(
                self.coords[:, axis] > dim_size
            ):
                raise IndexError(
                    f"Coordinate axis {axis} out of range [1, {dim_size}]"
                )

        # Convert 1-based xyz to 0-based linear indices (Fortran order)
        linear = (
            (self.coords[:, 0] - 1)
            + (self.coords[:, 1] - 1) * dims[0]
            + (self.coords[:, 2] - 1) * dims[0] * dims[1]
        )

        mask_indices = np.where(mask.ravel())[0]
        mask_set = set(mask_indices.tolist())

        cols: list[int] = []
        skipped = 0
        for lin_idx in linear:
            if lin_idx in mask_set:
                col = int(np.searchsorted(mask_indices, lin_idx))
                cols.append(col)
            else:
                skipped += 1

        if skipped > 0:
            import warnings

            warnings.warn(
                f"{skipped} voxel(s) outside the dataset mask were ignored",
                stacklevel=2,
            )

        if not cols:
            raise ValueError("No requested voxels are within the dataset mask")

        return np.array(cols, dtype=np.intp)


class SphereSelector(SeriesSelector):
    """Select voxels within a spherical region.

    Parameters
    ----------
    center : array-like of length 3
        Sphere center in 1-based voxel coordinates.
    radius : float
        Radius in voxel units.
    """

    def __init__(
        self,
        center: NDArray[np.floating[Any]] | list[float] | tuple[float, ...],
        radius: float,
    ) -> None:
        c = np.asarray(center, dtype=np.float64)
        if c.size != 3:
            raise ValueError("center must have length 3")
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.center = c
        self.radius = float(radius)

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dims = dataset.get_dims().spatial
        mask = dataset.get_mask().ravel()

        # Build coordinate grid (1-based)
        zz, yy, xx = np.meshgrid(
            np.arange(1, dims[2] + 1),
            np.arange(1, dims[1] + 1),
            np.arange(1, dims[0] + 1),
            indexing="ij",
        )
        coords = np.stack(
            [xx.ravel(), yy.ravel(), zz.ravel()], axis=1
        )  # shape (n_total, 3)

        dist = np.sqrt(np.sum((coords - self.center) ** 2, axis=1))
        in_sphere = dist <= self.radius

        # Intersect with mask
        mask_indices = np.where(mask)[0]
        sphere_indices = set(np.where(in_sphere)[0].tolist())

        cols: list[int] = []
        for col_idx, vox_idx in enumerate(mask_indices):
            if vox_idx in sphere_indices:
                cols.append(col_idx)

        if not cols:
            raise ValueError(
                "Spherical ROI does not overlap with dataset mask"
            )

        return np.array(cols, dtype=np.intp)


class MaskSelector(SeriesSelector):
    """Select voxels by a logical mask.

    Parameters
    ----------
    mask : ndarray of bool
        Boolean mask. Can be a flat vector matching either the full volume
        size or the masked voxel count.
    """

    def __init__(self, mask: NDArray[np.bool_]) -> None:
        self.mask = np.asarray(mask, dtype=np.bool_).ravel()

    def resolve_indices(self, dataset: FmriDataset) -> NDArray[np.intp]:
        dataset_mask = dataset.get_mask().ravel()
        n_volume = dataset_mask.size
        n_masked = int(dataset_mask.sum())

        if self.mask.size == n_volume:
            # Full-volume mask — intersect with dataset mask
            mask_indices = np.where(dataset_mask)[0]
            selection_indices = set(np.where(self.mask)[0].tolist())
            cols: list[int] = []
            for col_idx, vox_idx in enumerate(mask_indices):
                if vox_idx in selection_indices:
                    cols.append(col_idx)
        elif self.mask.size == n_masked:
            # Already in masked space
            cols = list(np.where(self.mask)[0])
        else:
            raise ValueError(
                f"Mask length ({self.mask.size}) does not match "
                f"volume size ({n_volume}) or masked size ({n_masked})"
            )

        if not cols:
            raise ValueError("Mask selector selected no voxels")

        return np.array(cols, dtype=np.intp)
