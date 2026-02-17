"""LatentDataset — specialized dataset for latent-space fMRI data.

Port of ``R/latent_dataset.R``.
"""

from __future__ import annotations

import warnings
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
        return self._latent_backend.get_loadings(
            components=components if components is None else np.asarray(components, dtype=np.intp)
        )

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

    def get_component_info(self) -> dict[str, Any]:
        """Return metadata about latent components."""
        return self._backend.get_metadata()

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return latent scores (time x components) with compatibility warning."""
        warnings.warn(
            "get_data() on latent_dataset returns latent scores, not voxel data. "
            "Use get_latent_scores() or reconstruct_voxels().",
            UserWarning,
            stacklevel=2,
        )
        return self.get_latent_scores(rows=rows, cols=cols)

    def get_data_matrix(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return latent scores as a dense matrix without a warning."""
        return self.get_latent_scores(rows=rows, cols=cols)

    def reconstruct_voxels(
        self,
        rows: NDArray[np.intp] | None = None,
        voxels: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Alias for latent to voxel-space reconstruction."""
        return self._latent_backend.reconstruct_voxels(rows=rows, voxels=voxels)


def latent_dataset(
    source: str | list[str],
    TR: float,  # noqa: N803
    run_length: int | Sequence[int] | None = None,
    base_path: str = ".",
    event_table: pd.DataFrame | None = None,
    censor: NDArray[np.intp] | None = None,
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
    censor : ndarray or None
        Optional censor vector for each time point.
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
    total_time = backend.get_dims().time
    if sum(run_length) != total_time:
        raise ValueError(
            f"sum(run_length) ({sum(run_length)}) must equal number of time points ({total_time})"
        )

    frame = SamplingFrame.create(blocklens=run_length, TR=TR)

    return LatentDataset(
        backend=backend,
        sampling_frame=frame,
        event_table=_ensure_event_table_unique(event_table),
        censor=np.asarray(censor, dtype=np.intp) if censor is not None else None,
    )
