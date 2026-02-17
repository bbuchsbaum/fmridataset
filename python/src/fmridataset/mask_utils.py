"""Mask conversion helpers.

Port of ``R/mask_standards.R``.  Enforces consistent mask representation
across all components.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mask_to_logical(mask: NDArray[np.generic]) -> NDArray[np.bool_]:
    """Convert any mask representation to a boolean vector.

    Parameters
    ----------
    mask : ndarray
        Numeric, boolean, or multi-dimensional mask array.

    Returns
    -------
    ndarray of bool
        Flat boolean vector where ``True`` = valid voxel.
    """
    return np.asarray(mask, dtype=np.bool_).ravel()


def mask_to_volume(
    mask_vec: NDArray[np.bool_],
    dims: tuple[int, int, int],
) -> NDArray[np.bool_]:
    """Reshape a flat boolean mask into a 3-D volume.

    Parameters
    ----------
    mask_vec : ndarray of bool
        Flat boolean vector.
    dims : tuple of (int, int, int)
        Spatial dimensions ``(x, y, z)``.

    Returns
    -------
    ndarray of bool
        3-D boolean array with shape *dims*.

    Raises
    ------
    ValueError
        If ``len(mask_vec) != prod(dims)``.
    """
    expected = dims[0] * dims[1] * dims[2]
    if mask_vec.size != expected:
        raise ValueError(
            f"Mask length ({mask_vec.size}) doesn't match spatial "
            f"dimensions {dims} (product={expected})"
        )
    return np.asarray(mask_vec, dtype=np.bool_).reshape(dims)
