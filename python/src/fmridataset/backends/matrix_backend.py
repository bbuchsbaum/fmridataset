"""In-memory matrix storage backend.

Port of ``R/matrix_backend.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import ConfigError


class MatrixBackend(StorageBackend):
    """Backend that wraps a plain NumPy matrix.

    Parameters
    ----------
    data_matrix : ndarray
        2-D array shaped ``(n_timepoints, n_voxels)``.
    mask : ndarray of bool or None
        Boolean mask of length ``n_voxels``.  Defaults to all-True.
    spatial_dims : tuple of int or None
        ``(x, y, z)`` grid dimensions.  ``prod(spatial_dims)`` must equal
        ``n_voxels``.  Defaults to ``(n_voxels, 1, 1)``.
    metadata : dict or None
        Arbitrary metadata dict.
    """

    def __init__(
        self,
        data_matrix: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_] | None = None,
        spatial_dims: tuple[int, int, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if data_matrix.ndim != 2:
            raise ConfigError(
                "data_matrix must be a 2-D array",
                parameter="data_matrix",
                value=data_matrix.ndim,
            )

        self._data = np.asarray(data_matrix, dtype=np.float64)
        n_voxels = self._data.shape[1]

        # Default mask
        if mask is None:
            self._mask = np.ones(n_voxels, dtype=np.bool_)
        else:
            self._mask = np.asarray(mask, dtype=np.bool_)

        if self._mask.shape != (n_voxels,):
            raise ConfigError(
                f"mask length ({self._mask.size}) must equal "
                f"number of columns ({n_voxels})",
                parameter="mask",
            )

        # Default spatial dims
        if spatial_dims is None:
            self._spatial_dims = (n_voxels, 1, 1)
        else:
            self._spatial_dims = spatial_dims

        if len(self._spatial_dims) != 3:
            raise ConfigError(
                "spatial_dims must have exactly 3 elements",
                parameter="spatial_dims",
                value=self._spatial_dims,
            )

        if int(np.prod(self._spatial_dims)) != n_voxels:
            raise ConfigError(
                f"prod(spatial_dims) ({int(np.prod(self._spatial_dims))}) "
                f"must equal number of voxels ({n_voxels})",
                parameter="spatial_dims",
            )

        self._metadata = metadata or {}

    # -- lifecycle (stateless) ---------------------------------------------

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    # -- introspection -----------------------------------------------------

    def get_dims(self) -> BackendDims:
        return BackendDims(
            spatial=self._spatial_dims,
            time=self._data.shape[0],
        )

    def get_mask(self) -> NDArray[np.bool_]:
        return self._mask.copy()

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        # First apply mask to get valid voxels
        data = self._data[:, self._mask]

        if rows is not None:
            data = data[rows, :]
        if cols is not None:
            data = data[:, cols]
        return data

    def get_metadata(self) -> dict[str, Any]:
        meta = {"format": "matrix"}
        meta.update(self._metadata)
        return meta
