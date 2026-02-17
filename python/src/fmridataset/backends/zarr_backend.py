"""Zarr array storage backend.

Port of ``R/zarr_backend.R``.  Requires ``zarr`` (optional dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import BackendIOError, ConfigError


class ZarrBackend(StorageBackend):
    """Backend for Zarr-format 4-D arrays.

    Expects the Zarr store to contain a single 4-D array with shape
    ``(x, y, z, time)``.

    Parameters
    ----------
    source : str or Path
        Path to the Zarr store directory.
    mask : ndarray of bool or None
        Optional explicit mask.  If ``None``, all voxels are assumed valid.
    preload : bool
        If True, eagerly load all data into memory.
    """

    def __init__(
        self,
        source: str | Path,
        mask: NDArray[np.bool_] | None = None,
        preload: bool = False,
    ) -> None:
        self._source = Path(source)
        self._external_mask = mask
        self._preload = preload

        self._zarr_array: Any = None
        self._data_cache: NDArray[np.floating[Any]] | None = None
        self._mask_vec: NDArray[np.bool_] | None = None
        self._dims: BackendDims | None = None
        self._is_open = False

    def open(self) -> None:
        try:
            import zarr as zarr_lib
        except ImportError as exc:
            raise ConfigError(
                "zarr is required for ZarrBackend. "
                "Install with: pip install zarr"
            ) from exc

        if not self._source.exists():
            raise BackendIOError(
                f"Zarr store not found: {self._source}",
                file=str(self._source),
                operation="open",
            )

        try:
            self._zarr_array = zarr_lib.open(str(self._source), mode="r")
        except Exception as exc:
            raise BackendIOError(
                f"Failed to open Zarr store: {exc}",
                file=str(self._source),
                operation="open",
            ) from exc

        shape = self._zarr_array.shape
        if len(shape) != 4:
            raise ConfigError(
                f"Expected 4D array, got {len(shape)}D",
                parameter="source",
                value=str(self._source),
            )

        spatial = (int(shape[0]), int(shape[1]), int(shape[2]))
        self._dims = BackendDims(spatial=spatial, time=int(shape[3]))

        n_voxels = int(np.prod(spatial))
        if self._external_mask is not None:
            self._mask_vec = np.asarray(self._external_mask, dtype=np.bool_).ravel()
            if self._mask_vec.size != n_voxels:
                raise ConfigError(
                    f"mask length ({self._mask_vec.size}) != "
                    f"prod(spatial) ({n_voxels})",
                    parameter="mask",
                )
        else:
            self._mask_vec = np.ones(n_voxels, dtype=np.bool_)

        if self._preload:
            raw = np.asarray(self._zarr_array, dtype=np.float64)
            n_time = raw.shape[3]
            flat = raw.reshape(-1, n_time).T  # (time, voxels_flat)
            voxel_idx = np.where(self._mask_vec)[0]
            self._data_cache = flat[:, voxel_idx]

        self._is_open = True

    def close(self) -> None:
        self._zarr_array = None
        self._data_cache = None
        self._is_open = False

    def get_dims(self) -> BackendDims:
        if self._dims is None:
            raise BackendIOError("Backend not opened", operation="get_dims")
        return self._dims

    def get_mask(self) -> NDArray[np.bool_]:
        if self._mask_vec is None:
            raise BackendIOError("Backend not opened", operation="get_mask")
        return self._mask_vec.copy()

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        if self._data_cache is not None:
            data = self._data_cache
        else:
            if self._zarr_array is None:
                raise BackendIOError("Backend not opened", operation="get_data")
            raw = np.asarray(self._zarr_array, dtype=np.float64)
            n_time = raw.shape[3]
            flat = raw.reshape(-1, n_time).T
            assert self._mask_vec is not None
            voxel_idx = np.where(self._mask_vec)[0]
            data = flat[:, voxel_idx]

        if rows is not None:
            data = data[rows, :]
        if cols is not None:
            data = data[:, cols]
        return data

    def get_metadata(self) -> dict[str, Any]:
        return {"format": "zarr", "storage_format": "zarr"}
