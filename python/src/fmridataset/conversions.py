"""Type conversions between dataset representations.

Port of ``R/conversions.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import FmriDataset, MatrixDataset
from .dataset_constructors import matrix_dataset


def to_matrix_dataset(dataset: FmriDataset) -> MatrixDataset:
    """Convert any :class:`FmriDataset` to a :class:`MatrixDataset`.

    If *dataset* is already a :class:`MatrixDataset`, return it unchanged.
    Otherwise, materialise the full data matrix via the backend and wrap it.
    """
    if isinstance(dataset, MatrixDataset):
        return dataset

    datamat = dataset.get_data_matrix()
    return matrix_dataset(
        datamat=datamat,
        TR=dataset.TR,
        run_length=list(dataset.blocklens),
        event_table=dataset.event_table,
    )
