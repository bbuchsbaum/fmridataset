"""HDF5 storage backend.

Reads the fmristore HDF5 layout directly using ``h5py`` — does **not**
depend on the R fmristore package.

Port of ``R/h5_backend.R``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import BackendIOError, ConfigError


class H5Backend(StorageBackend):
    """Backend for HDF5-format fMRI data (fmristore schema).

    Parameters
    ----------
    source : list of str/Path
        Paths to HDF5 data files.
    mask_source : str or Path
        Path to the HDF5 file containing the mask.
    mask_dataset : str
        HDF5 dataset path for the mask (default ``"data/elements"``).
    data_dataset : str
        HDF5 dataset path for the data (default ``"data"``).
    preload : bool
        Eagerly load everything into memory.
    """

    def __init__(
        self,
        source: list[str] | list[Path] | str | Path,
        mask_source: str | Path,
        mask_dataset: str = "data/elements",
        data_dataset: str = "data",
        preload: bool = False,
    ) -> None:
        sources: list[str | Path] = [source] if isinstance(source, (str, Path)) else list(source)
        self._source = [Path(s) for s in sources]
        self._mask_source = Path(mask_source)
        self._mask_dataset = mask_dataset
        self._data_dataset = data_dataset
        self._preload = preload

        self._data: NDArray[np.floating[Any]] | None = None
        self._mask_vec: NDArray[np.bool_] | None = None
        self._dims: BackendDims | None = None
        self._metadata: dict[str, Any] | None = None
        self._is_open = False

    def open(self) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ConfigError(
                "h5py is required for H5Backend. "
                "Install with: pip install h5py"
            ) from exc

        # Read mask from first data file or separate mask file
        if not self._mask_source.exists():
            raise BackendIOError(
                f"Mask file not found: {self._mask_source}",
                file=str(self._mask_source),
                operation="open",
            )

        with h5py.File(str(self._mask_source), "r") as f:
            if self._mask_dataset in f:
                mask_data = np.asarray(f[self._mask_dataset])
                self._mask_vec = (mask_data.ravel() > 0).astype(np.bool_)
            else:
                raise BackendIOError(
                    f"Mask dataset '{self._mask_dataset}' not found in {self._mask_source}",
                    file=str(self._mask_source),
                    operation="open",
                )

        # Read dimensions from first data file
        total_time = 0
        spatial_shape: tuple[int, int, int] | None = None
        for src in self._source:
            if not src.exists():
                raise BackendIOError(
                    f"Data file not found: {src}",
                    file=str(src),
                    operation="open",
                )
            with h5py.File(str(src), "r") as f:
                if self._data_dataset not in f:
                    raise BackendIOError(
                        f"Data dataset '{self._data_dataset}' not found in {src}",
                        file=str(src),
                        operation="open",
                    )
                shape = f[self._data_dataset].shape
                if len(shape) == 4:
                    if spatial_shape is None:
                        spatial_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
                    total_time += shape[3]
                elif len(shape) == 2:
                    # Already (time, voxels)
                    total_time += shape[0]
                    if spatial_shape is None:
                        # Infer from mask length
                        n_vox = self._mask_vec.size
                        spatial_shape = (n_vox, 1, 1)

        if spatial_shape is None:
            raise BackendIOError(
                "Could not determine spatial dimensions",
                operation="open",
            )

        self._dims = BackendDims(spatial=spatial_shape, time=total_time)
        self._metadata = {"format": "h5"}

        if self._preload:
            self._load_data()

        self._is_open = True

    def _load_data(self) -> None:
        import h5py

        assert self._mask_vec is not None
        voxel_idx = np.where(self._mask_vec)[0]
        parts: list[NDArray[np.floating[Any]]] = []
        for src in self._source:
            with h5py.File(str(src), "r") as f:
                raw = np.asarray(f[self._data_dataset], dtype=np.float64)
                if raw.ndim == 4:
                    n_time = raw.shape[3]
                    flat = raw.reshape(-1, n_time).T  # (time, voxels_flat)
                    parts.append(flat[:, voxel_idx])
                elif raw.ndim == 2:
                    # Already (time, voxels)
                    parts.append(raw[:, voxel_idx] if voxel_idx.max() < raw.shape[1] else raw)
        self._data = np.concatenate(parts, axis=0)

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
            self._load_data()
        assert self._data is not None

        data = self._data
        if rows is not None:
            data = data[rows, :]
        if cols is not None:
            data = data[:, cols]
        return data

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata) if self._metadata else {}
