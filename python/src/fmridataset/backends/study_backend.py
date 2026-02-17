"""Multi-subject composite storage backend.

Port of ``R/study_backend.R``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import ConfigError


class StudyBackend(StorageBackend):
    """Composite backend that lazily combines multiple subject backends.

    Parameters
    ----------
    backends : list of StorageBackend
        One backend per subject.
    subject_ids : list of str/int or None
        Identifiers for each subject. Defaults to 1..N.
    strict : str
        Mask validation mode: ``"identical"`` (default) or ``"intersect"``.
    """

    def __init__(
        self,
        backends: Sequence[StorageBackend],
        subject_ids: Sequence[Any] | None = None,
        strict: str = "identical",
    ) -> None:
        if not backends:
            raise ConfigError("backends must be a non-empty list")

        self._backends = list(backends)
        self._subject_ids = (
            list(subject_ids) if subject_ids is not None
            else list(range(1, len(backends) + 1))
        )
        self._strict = strict

        if len(self._subject_ids) != len(self._backends):
            raise ConfigError("subject_ids must match length of backends")

        # Validate spatial dimensions match
        dims_list = [b.get_dims() for b in self._backends]
        ref_spatial = dims_list[0].spatial
        for i, d in enumerate(dims_list[1:], 1):
            if d.spatial != ref_spatial:
                raise ConfigError(
                    f"Spatial dimensions of backend {i} {d.spatial} "
                    f"do not match reference {ref_spatial}"
                )

        # Validate and combine masks
        masks = [b.get_mask() for b in self._backends]
        ref_mask = masks[0]
        if strict == "identical":
            for i, m in enumerate(masks[1:], 1):
                if not np.array_equal(m, ref_mask):
                    raise ConfigError(
                        f"Mask of backend {i} differs from reference"
                    )
            combined_mask = ref_mask
        elif strict == "intersect":
            for i, m in enumerate(masks[1:], 1):
                overlap = np.sum(m & ref_mask) / len(ref_mask)
                if overlap < 0.95:
                    raise ConfigError(
                        f"Mask overlap ({overlap:.2%}) < 95% for backend {i}"
                    )
            combined_mask = masks[0].copy()
            for m in masks[1:]:
                combined_mask &= m
        else:
            raise ConfigError(f"Unknown strict setting: {strict}")

        time_dims = [d.time for d in dims_list]
        total_time = sum(time_dims)

        self._dims = BackendDims(spatial=ref_spatial, time=total_time)
        self._mask = combined_mask
        self._time_dims = time_dims
        self._subject_boundaries = np.array(
            [0] + list(np.cumsum(time_dims)), dtype=np.intp
        )

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        for b in self._backends:
            b.open()

    def close(self) -> None:
        for b in self._backends:
            b.close()

    # -- introspection -----------------------------------------------------

    def get_dims(self) -> BackendDims:
        return self._dims

    def get_mask(self) -> NDArray[np.bool_]:
        return self._mask.copy()

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        total_time = self._dims.time
        n_vox = int(self._mask.sum())

        if rows is None:
            rows = np.arange(total_time, dtype=np.intp)
        if cols is None:
            cols = np.arange(n_vox, dtype=np.intp)

        rows = np.asarray(rows, dtype=np.intp)
        cols = np.asarray(cols, dtype=np.intp)

        # Sort rows for efficient subject-boundary lookup
        order = np.argsort(rows)
        sorted_rows = rows[order]
        result = np.empty((len(rows), len(cols)), dtype=np.float64)

        for s, backend in enumerate(self._backends):
            start = int(self._subject_boundaries[s])
            end = int(self._subject_boundaries[s + 1])
            mask = (sorted_rows >= start) & (sorted_rows < end)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            local_rows = sorted_rows[idx] - start
            subj_data = backend.get_data(rows=local_rows, cols=cols)
            result[idx, :] = subj_data

        # Unsort
        out = np.empty_like(result)
        out[order, :] = result
        return out

    def get_metadata(self) -> dict[str, Any]:
        first_meta = self._backends[0].get_metadata() if self._backends else {}
        first_meta["format"] = "study"
        first_meta["storage_format"] = "study"
        first_meta["n_subjects"] = len(self._backends)
        first_meta["subject_ids"] = self._subject_ids
        return first_meta

    # -- study-specific API ------------------------------------------------

    @property
    def subject_ids(self) -> list[Any]:
        return self._subject_ids

    @property
    def subject_boundaries(self) -> NDArray[np.intp]:
        return self._subject_boundaries
