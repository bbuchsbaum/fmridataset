"""NIfTI storage backend using nibabel.

Port of ``R/nifti_backend.R``.  Requires ``nibabel`` (optional dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import BackendIOError, ConfigError


class NiftiBackend(StorageBackend):
    """Backend for NIfTI-format neuroimaging files.

    Parameters
    ----------
    source : list of str or Path
        Paths to NIfTI images.
    mask_source : str or Path
        Path to the binary mask image.
    preload : bool
        If True, eagerly load all data into memory.
    """

    def __init__(
        self,
        source: list[str] | list[Path] | str | Path,
        mask_source: str | Path,
        preload: bool = False,
    ) -> None:
        sources: list[str | Path] = [source] if isinstance(source, (str, Path)) else list(source)
        self._source = [Path(s) for s in sources]
        self._mask_source = Path(mask_source)
        self._preload = preload

        self._data: NDArray[np.floating[Any]] | None = None
        self._mask_vec: NDArray[np.bool_] | None = None
        self._dims: BackendDims | None = None
        self._metadata: dict[str, Any] | None = None
        self._is_open = False

    def open(self) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ConfigError(
                "nibabel is required for NiftiBackend. "
                "Install with: pip install nibabel"
            ) from exc

        # Read mask
        if not self._mask_source.exists():
            raise BackendIOError(
                f"Mask file not found: {self._mask_source}",
                file=str(self._mask_source),
                operation="open",
            )
        mask_img: Any = nib.load(str(self._mask_source))  # type: ignore[attr-defined]
        mask_data = np.asarray(mask_img.dataobj)
        self._mask_vec = (mask_data.ravel() > 0).astype(np.bool_)
        spatial_shape = mask_data.shape[:3]

        # Read headers to get time dimension
        total_time = 0
        for src in self._source:
            if not src.exists():
                raise BackendIOError(
                    f"Source file not found: {src}",
                    file=str(src),
                    operation="open",
                )
            hdr: Any = nib.load(str(src))  # type: ignore[attr-defined]
            shape = hdr.shape
            total_time += shape[3] if len(shape) > 3 else 1

        self._dims = BackendDims(
            spatial=(int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2])),
            time=total_time,
        )

        # Store affine from mask
        self._metadata = {
            "format": "nifti",
            "affine": np.array(mask_img.affine),
            "voxel_dims": np.array(mask_img.header.get_zooms()[:3]),
        }

        if self._preload:
            self._load_data()

        self._is_open = True

    def _load_data(self) -> None:
        import nibabel as nib

        assert self._mask_vec is not None
        voxel_idx = np.where(self._mask_vec)[0]
        parts: list[NDArray[np.floating[Any]]] = []
        for src in self._source:
            img: Any = nib.load(str(src))  # type: ignore[attr-defined]
            vol4d = np.asarray(img.dataobj, dtype=np.float64)
            if vol4d.ndim == 3:
                vol4d = vol4d[..., np.newaxis]
            n_time = vol4d.shape[3]
            flat = vol4d.reshape(-1, n_time).T  # (time, voxels_flat)
            parts.append(flat[:, voxel_idx])
        self._data = np.concatenate(parts, axis=0)

    def close(self) -> None:
        self._data = None
        self._is_open = False

    def get_dims(self) -> BackendDims:
        if self._dims is None:
            raise BackendIOError(
                "Backend not opened", operation="get_dims"
            )
        return self._dims

    def get_mask(self) -> NDArray[np.bool_]:
        if self._mask_vec is None:
            raise BackendIOError(
                "Backend not opened", operation="get_mask"
            )
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
