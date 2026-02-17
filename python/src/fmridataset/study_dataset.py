"""StudyDataset — multi-subject composite dataset.

Port of ``fmri_study_dataset()`` from ``R/dataset_constructors.R``.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .backends.matrix_backend import MatrixBackend
from .backends.study_backend import StudyBackend
from .dataset import FmriDataset
from .errors import ConfigError
from .sampling_frame import SamplingFrame


class StudyDataset(FmriDataset):
    """Dataset combining multiple subjects.

    In addition to the standard :class:`FmriDataset` API, exposes
    ``subject_ids``.
    """

    def __init__(
        self,
        backend: StudyBackend,
        sampling_frame: SamplingFrame,
        subject_ids: list[Any],
        event_table: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            sampling_frame=sampling_frame,
            event_table=event_table,
        )
        self._subject_ids = subject_ids

    @property
    def subject_ids(self) -> list[Any]:
        return self._subject_ids

    def __repr__(self) -> str:
        return (
            f"<StudyDataset "
            f"subjects={len(self._subject_ids)} "
            f"time={self.n_timepoints} "
            f"runs={self.n_runs} TR={self.TR}>"
        )


def study_dataset(
    datasets: Sequence[FmriDataset],
    subject_ids: Sequence[Any] | None = None,
    strict: str = "identical",
) -> StudyDataset:
    """Combine multiple :class:`FmriDataset` objects into a study.

    Parameters
    ----------
    datasets : sequence of FmriDataset
        One dataset per subject.
    subject_ids : sequence or None
        Subject identifiers (defaults to 1..N).
    strict : str
        Mask validation: ``"identical"`` or ``"intersect"``.
    """
    if not datasets:
        raise ConfigError("datasets must be non-empty")

    if subject_ids is None:
        subject_ids_list = list(range(1, len(datasets) + 1))
    else:
        subject_ids_list = list(subject_ids)

    if len(subject_ids_list) != len(datasets):
        raise ConfigError("subject_ids must match length of datasets")

    # Check TR consistency
    trs = [ds.TR for ds in datasets]
    if len(set(round(t, 10) for t in trs)) > 1:
        raise ConfigError("All datasets must have the same TR")

    # Extract or create backends
    backends = []
    for ds in datasets:
        if hasattr(ds, '_backend'):
            backends.append(ds._backend)
        elif hasattr(ds, '_datamat'):
            # MatrixDataset fallback
            backends.append(MatrixBackend(data_matrix=ds._datamat))
        else:
            raise ConfigError("Cannot extract backend from dataset")

    sb = StudyBackend(backends=backends, subject_ids=subject_ids_list, strict=strict)

    # Combine run lengths and events
    all_blocklens: list[int] = []
    all_events: list[pd.DataFrame] = []
    for ds, sid in zip(datasets, subject_ids_list):
        all_blocklens.extend(ds.blocklens)
        et = ds.event_table.copy()
        if len(et) > 0:
            et["subject_id"] = sid

            if "run_id" not in et.columns:
                if ds.n_runs == 1:
                    et["run_id"] = np.full(len(et), 1, dtype=int)
                else:
                    warnings.warn(
                        f"event_table for subject '{sid}' has no 'run_id' column and "
                        f"dataset has {ds.n_runs} runs. Please add a 'run_id' column to "
                        "avoid ambiguity.",
                        UserWarning,
                        stacklevel=2,
                    )
                    et["run_id"] = np.resize(np.arange(1, ds.n_runs + 1), len(et))

        all_events.append(et)

    combined_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    frame = SamplingFrame.create(blocklens=all_blocklens, TR=trs[0])

    return StudyDataset(
        backend=sb,
        sampling_frame=frame,
        subject_ids=subject_ids_list,
        event_table=combined_events,
    )
