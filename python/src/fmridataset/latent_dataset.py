"""LatentDataset — specialized dataset for latent-space fMRI data.

Port of ``R/latent_dataset.R``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .backends.latent_backend import LatentBackend
from .dataset import FmriDataset
from .sampling_frame import SamplingFrame


class LatentDataset(FmriDataset):
    """Dataset wrapping a :class:`LatentBackend`.

    Provides additional methods for accessing the latent decomposition
    (scores, loadings, reconstruction).
    """

    @property
    def _latent_backend(self) -> LatentBackend:
        assert isinstance(self._backend, LatentBackend)
        return self._backend

    def get_latent_scores(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return the latent scores (time x components)."""
        return self._latent_backend.get_data(rows=rows, cols=cols)

    def get_spatial_loadings(self) -> NDArray[np.floating[Any]]:
        """Return the spatial loadings matrix (voxels x components)."""
        lb = self._latent_backend
        if lb._loadings is None:
            raise RuntimeError("Backend not opened")
        return lb._loadings.copy()

    def __repr__(self) -> str:
        meta = self._backend.get_metadata()
        return (
            f"<LatentDataset "
            f"time={self.n_timepoints} "
            f"components={meta.get('n_components', '?')} "
            f"runs={self.n_runs} TR={self.TR}>"
        )


def latent_dataset(
    source: str | list[str],
    TR: float,  # noqa: N803
    run_length: int | Sequence[int],
    event_table: pd.DataFrame | None = None,
    preload: bool = False,
) -> LatentDataset:
    """Create a :class:`LatentDataset` from HDF5 latent-decomposition files.

    Parameters
    ----------
    source : str or list of str
        Path(s) to ``.lv.h5`` files.
    TR : float
        Repetition time in seconds.
    run_length : int or sequence of int
        Number of time-points per run.
    event_table : DataFrame or None
        Optional event information.
    preload : bool
        Eagerly materialise the reconstruction.
    """
    if isinstance(run_length, (int, np.integer)):
        run_length = [int(run_length)]
    else:
        run_length = [int(r) for r in run_length]

    backend = LatentBackend(source=source, preload=preload)
    backend.open()

    frame = SamplingFrame.create(blocklens=run_length, TR=TR)

    return LatentDataset(
        backend=backend,
        sampling_frame=frame,
        event_table=event_table,
    )
