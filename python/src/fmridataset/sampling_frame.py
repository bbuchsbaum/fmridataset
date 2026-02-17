"""SamplingFrame: temporal structure representation for fMRI acquisitions.

Direct port of ``fmrihrf::sampling_frame`` plus the adapter methods from
``R/sampling_frame_adapters.R``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SamplingFrame:
    """Immutable description of the temporal structure of an fMRI acquisition.

    Parameters
    ----------
    blocklens : tuple[int, ...]
        Number of time-points (volumes) in each run/block.
    TR : float
        Repetition time in seconds.
    """

    blocklens: tuple[int, ...]
    TR: float  # noqa: N815 – matches R naming

    def __post_init__(self) -> None:
        if not self.blocklens:
            raise ValueError("blocklens must be non-empty")
        if any(b <= 0 for b in self.blocklens):
            raise ValueError("all blocklens must be positive integers")
        if self.TR <= 0:
            raise ValueError("TR must be positive")

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        blocklens: Sequence[int],
        TR: float,  # noqa: N803
    ) -> SamplingFrame:
        """Create a SamplingFrame from a sequence of block lengths and TR."""
        return cls(blocklens=tuple(int(b) for b in blocklens), TR=float(TR))

    # ------------------------------------------------------------------
    # Properties – mirror R generic methods
    # ------------------------------------------------------------------

    @property
    def n_runs(self) -> int:
        """Number of runs/blocks."""
        return len(self.blocklens)

    @property
    def n_timepoints(self) -> int:
        """Total number of time-points across all runs."""
        return sum(self.blocklens)

    @property
    def blockids(self) -> NDArray[np.intp]:
        """1-based run id for every time-point (matches R ``rep(seq_along(...))``).

        Returns an array of length ``n_timepoints`` where each element is the
        1-based index of the run that time-point belongs to.
        """
        return np.repeat(
            np.arange(1, self.n_runs + 1),
            self.blocklens,
        )

    @property
    def samples(self) -> NDArray[np.intp]:
        """1-based sample indices (1 .. n_timepoints)."""
        return np.arange(1, self.n_timepoints + 1)

    @property
    def run_durations(self) -> NDArray[np.float64]:
        """Duration of each run in seconds (blocklens * TR)."""
        return np.array(self.blocklens, dtype=np.float64) * self.TR

    @property
    def total_duration(self) -> float:
        """Total acquisition duration in seconds."""
        return float(np.sum(self.run_durations))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def run_indices(self, run: int) -> NDArray[np.intp]:
        """Return 0-based row indices for the given 1-based *run* number."""
        if run < 1 or run > self.n_runs:
            raise ValueError(
                f"run must be between 1 and {self.n_runs}, got {run}"
            )
        start = sum(self.blocklens[: run - 1])
        end = start + self.blocklens[run - 1]
        return np.arange(start, end)

    def __repr__(self) -> str:
        return (
            f"SamplingFrame(n_runs={self.n_runs}, "
            f"n_timepoints={self.n_timepoints}, TR={self.TR})"
        )
