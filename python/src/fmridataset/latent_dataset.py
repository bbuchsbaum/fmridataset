"""LatentDataset — specialized dataset for latent-space fMRI data.

Port of ``R/latent_dataset.R``.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .backends.latent_backend import LatentBackend
from .dataset import FmriDataset
from .sampling_frame import SamplingFrame
from .dataset_constructors import _coerce_run_length, _ensure_event_table_unique


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

    def get_spatial_loadings(
        self,
        components: NDArray[np.intp] | Sequence[int] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return spatial loadings (voxels x components).

        Parameters
        ----------
        components:
            Optional component indices to subset columns.
        """
        lb = self._latent_backend
        if lb._loadings is None:
            raise RuntimeError("Backend not opened")
        loadings = lb._loadings.copy()
        if components is None:
            return loadings
        comps = np.asarray(components, dtype=np.intp)
        return loadings[:, comps]

    def __repr__(self) -> str:
        meta = self._backend.get_metadata()
        return (
            f"<LatentDataset "
            f"time={self.n_timepoints} "
            f"components={meta.get('n_components', '?')} "
            f"runs={self.n_runs} TR={self.TR}>"
        )

    def get_mask(self) -> NDArray[np.bool_]:
        """Return mask for latent components."""
        meta = self._backend.get_metadata()
        n_components = int(meta.get("n_components", 0))
        return np.ones(n_components, dtype=np.bool_)


def latent_dataset(
    source: str | list[str],
    TR: float,  # noqa: N803
    run_length: int | Sequence[int] | None = None,
    base_path: str = ".",
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
    base_path : str
        Base directory for relative source files.
    event_table : DataFrame or None
        Optional event information.
    preload : bool
        Eagerly materialise the reconstruction.
    """
    if isinstance(source, (str, Path)):
        source_paths = [Path(base_path) / Path(source)]
    else:
        source_paths = []
        for src in source:
            src_path = Path(src)
            if src_path.is_absolute():
                source_paths.append(src_path)
            else:
                source_paths.append(Path(base_path) / src_path)

    backend = LatentBackend(source=source_paths, preload=preload)
    backend.open()

    if run_length is None:
        run_length = backend.run_lengths
    else:
        if isinstance(run_length, (int, np.integer)):
            if int(run_length) == 0:
                run_length = backend.run_lengths
            else:
                run_length = [int(run_length)]
        else:
            run_length_seq = list(run_length)
            if len(run_length_seq) == 0 or sum(run_length_seq) == 0:
                run_length = backend.run_lengths
            else:
                run_length = run_length_seq

    run_length = _coerce_run_length(run_length)

    frame = SamplingFrame.create(blocklens=run_length, TR=TR)

    return LatentDataset(
        backend=backend,
        sampling_frame=frame,
        event_table=_ensure_event_table_unique(event_table),
    )
