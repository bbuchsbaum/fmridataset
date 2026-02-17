"""StorageBackend ABC and BackendDims dataclass.

Defines the contract that all storage backends must implement.  Direct port
of ``R/storage_backend.R``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigError


@dataclass(frozen=True)
class BackendDims:
    """Dimensions reported by a storage backend.

    Parameters
    ----------
    spatial : tuple[int, int, int]
        (x, y, z) voxel grid dimensions.
    time : int
        Number of time-points.
    """

    spatial: tuple[int, int, int]
    time: int

    def __post_init__(self) -> None:
        if len(self.spatial) != 3:
            raise ConfigError(
                "spatial must have exactly 3 elements",
                parameter="spatial",
                value=self.spatial,
            )
        if any(s < 1 for s in self.spatial):
            raise ConfigError(
                "all spatial dimensions must be >= 1",
                parameter="spatial",
                value=self.spatial,
            )
        if self.time < 1:
            raise ConfigError(
                "time dimension must be >= 1",
                parameter="time",
                value=self.time,
            )

    @property
    def n_spatial(self) -> int:
        """Total number of spatial elements (product of spatial dims)."""
        return int(np.prod(self.spatial))


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All backends must implement the six abstract methods below.
    Data orientation is always **timepoints x voxels**.
    """

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Acquire any necessary resources (file handles, connections)."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    def __enter__(self) -> StorageBackend:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- introspection -----------------------------------------------------

    @abstractmethod
    def get_dims(self) -> BackendDims:
        """Return the dimensions of the stored data."""

    @abstractmethod
    def get_mask(self) -> NDArray[np.bool_]:
        """Return a flat boolean mask of length ``prod(spatial_dims)``.

        Invariants
        ----------
        * ``len(mask) == prod(get_dims().spatial)``
        * ``mask.sum() > 0``
        * No ``NaN`` values.
        """

    @abstractmethod
    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Read data in **timepoints x voxels** orientation.

        Parameters
        ----------
        rows : array of int or None
            Time-point indices to read. ``None`` means all.
        cols : array of int or None
            Voxel indices (within the mask) to read. ``None`` means all.

        Returns
        -------
        ndarray
            2-D array shaped ``(len(rows), len(cols))``.
        """

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return backend-specific metadata (affine, voxel_dims, …)."""

    # -- validation --------------------------------------------------------

    def validate(self) -> bool:
        """Check that this backend satisfies the contract invariants.

        Returns ``True`` on success; raises :class:`ConfigError` on failure.
        """
        dims = self.get_dims()

        if not isinstance(dims, BackendDims):
            raise ConfigError(
                "get_dims() must return a BackendDims instance",
            )

        mask = self.get_mask()

        if mask.dtype != np.bool_:
            raise ConfigError("get_mask() must return a boolean array")

        expected = int(np.prod(dims.spatial))
        if mask.shape != (expected,):
            raise ConfigError(
                f"mask length ({mask.size}) must equal "
                f"prod(spatial dims) ({expected})",
            )

        if np.any(np.isnan(mask.astype(float))):
            raise ConfigError("mask must not contain NaN values")

        if mask.sum() == 0:
            raise ConfigError(
                "mask must contain at least one True value",
            )

        return True
