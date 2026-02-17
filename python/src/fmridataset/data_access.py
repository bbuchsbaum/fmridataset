"""Convenience free-functions for data access.

Port of ``R/data_access.R`` helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dataset import FmriDataset


def get_data(
    dataset: FmriDataset,
    rows: NDArray[np.intp] | None = None,
    cols: NDArray[np.intp] | None = None,
) -> NDArray[np.floating[Any]]:
    """Extract data matrix from *dataset*."""
    return dataset.get_data(rows=rows, cols=cols)


def get_data_matrix(
    dataset: FmriDataset,
    rows: NDArray[np.intp] | None = None,
    cols: NDArray[np.intp] | None = None,
) -> NDArray[np.floating[Any]]:
    """Extract data as a 2-D ndarray (timepoints x voxels)."""
    return dataset.get_data_matrix(rows=rows, cols=cols)


def get_mask(dataset: FmriDataset) -> NDArray[np.bool_]:
    """Extract the boolean mask from *dataset*."""
    return dataset.get_mask()
