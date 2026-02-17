"""Latent-space storage backend.

Reads HDF5 files containing a latent decomposition:
    basis    — (time, n_components)
    loadings — (n_voxels, n_components), possibly sparse
    offset   — (n_voxels,), optional

Reconstruction: ``data = basis @ loadings.T + offset``

Port of the latent_backend portion of ``R/latent_dataset.R``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import BackendIOError, ConfigError


class LatentBackend(StorageBackend):
    """Backend for latent-decomposition HDF5 files.

    Parameters
    ----------
    source : list of str/Path
        HDF5 files each containing ``basis``, ``loadings``, and optionally
        ``offset`` datasets.
    preload : bool
        Eagerly materialise the full reconstruction into memory.
    """

    def __init__(
        self,
        source: list[str] | list[Path] | str | Path,
        preload: bool = False,
    ) -> None:
        sources: list[str | Path] = [source] if isinstance(source, (str, Path)) else list(source)
        self._source = [Path(s) for s in sources]
        self._preload = preload

        self._basis_parts: list[NDArray[np.floating[Any]]] = []
        self._loadings: NDArray[np.floating[Any]] | None = None
        self._offset: NDArray[np.floating[Any]] | None = None
        self._mask_vec: NDArray[np.bool_] | None = None
        self._dims: BackendDims | None = None
        self._data: NDArray[np.floating[Any]] | None = None
        self._is_open = False

    def open(self) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ConfigError(
                "h5py is required for LatentBackend. "
                "Install with: pip install h5py"
            ) from exc

        self._basis_parts = []
        total_time = 0
        n_voxels: int | None = None

        for src in self._source:
            if not src.exists():
                raise BackendIOError(
                    f"Source file not found: {src}",
                    file=str(src),
                    operation="open",
                )
            with h5py.File(str(src), "r") as f:
                if "basis" not in f:
                    raise BackendIOError(
                        f"'basis' dataset not found in {src}",
                        file=str(src),
                        operation="open",
                    )
                basis = np.asarray(f["basis"], dtype=np.float64)
                self._basis_parts.append(basis)
                total_time += basis.shape[0]

                if self._loadings is None and "loadings" in f:
                    self._loadings = np.asarray(f["loadings"], dtype=np.float64)
                    n_voxels = self._loadings.shape[0]

                if self._offset is None and "offset" in f:
                    self._offset = np.asarray(f["offset"], dtype=np.float64)

        if self._loadings is None:
            raise BackendIOError(
                "No 'loadings' dataset found in any source file",
                operation="open",
            )
        if n_voxels is None:
            raise BackendIOError(
                "Could not determine n_voxels",
                operation="open",
            )

        self._dims = BackendDims(
            spatial=(n_voxels, 1, 1),
            time=total_time,
        )

        # Mask: all voxels valid
        self._mask_vec = np.ones(n_voxels, dtype=np.bool_)

        if self._preload:
            self._reconstruct()

        self._is_open = True

    def _reconstruct(self) -> None:
        """Materialise ``basis @ loadings.T + offset``."""
        basis = np.concatenate(self._basis_parts, axis=0)
        assert self._loadings is not None
        data = basis @ self._loadings.T
        if self._offset is not None:
            data += self._offset[np.newaxis, :]
        self._data = data

    def close(self) -> None:
        self._data = None
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
        if self._data is None:
            self._reconstruct()
        assert self._data is not None

        data = self._data
        if rows is not None:
            data = data[rows, :]
        if cols is not None:
            data = data[:, cols]
        return data

    def get_metadata(self) -> dict[str, Any]:
        n_components = self._loadings.shape[1] if self._loadings is not None else 0
        return {
            "format": "latent_h5",
            "n_components": n_components,
            "has_offset": self._offset is not None,
        }
