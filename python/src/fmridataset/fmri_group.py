"""FmriGroup — multi-subject group container.

Port of ``fmri_group()``, ``iter_subjects()``, ``group_map()``,
``group_reduce()``, ``filter_subjects()``, ``sample_subjects()``
from R.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Iterator, Literal, Sequence, TypeVar

import numpy as np
import pandas as pd

from .dataset import FmriDataset
from .errors import ConfigError

T = TypeVar("T")
U = TypeVar("U")


class FmriGroup:
    """Group container wrapping a DataFrame of per-subject datasets.

    Parameters
    ----------
    subjects : DataFrame
        One row per subject. Must contain *id_col* and *dataset_col*.
    id_col : str
        Name of the subject-identifier column.
    dataset_col : str
        Name of the list-column storing :class:`FmriDataset` objects.
    space : str or None
        Nominal common space label (e.g. ``"MNI152"``).
    mask_strategy : str
        One of ``"subject_specific"``, ``"intersect"``, ``"union"``.
    """

    def __init__(
        self,
        subjects: pd.DataFrame,
        id_col: str,
        dataset_col: str = "dataset",
        space: str | None = None,
        mask_strategy: Literal[
            "subject_specific", "intersect", "union"
        ] = "subject_specific",
    ) -> None:
        if not isinstance(subjects, pd.DataFrame):
            raise ConfigError("subjects must be a DataFrame")
        if id_col not in subjects.columns:
            raise ConfigError(f"id column '{id_col}' not found in subjects")
        if dataset_col not in subjects.columns:
            raise ConfigError(
                f"dataset column '{dataset_col}' not found in subjects"
            )

        # Validate dataset column contains datasets
        for i, val in enumerate(subjects[dataset_col]):
            if val is None:
                warnings.warn(
                    f"Subject at row {i} has None in '{dataset_col}'",
                    stacklevel=2,
                )

        self._subjects = subjects.copy()
        self._id_col = id_col
        self._dataset_col = dataset_col
        self._space = space
        self._mask_strategy = mask_strategy

    @property
    def subjects(self) -> pd.DataFrame:
        """The underlying subjects DataFrame."""
        return self._subjects

    @subjects.setter
    def subjects(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ConfigError("subjects must be a DataFrame")
        if self._dataset_col not in value.columns:
            raise ConfigError(
                f"Replacement subjects must have column '{self._dataset_col}'"
            )
        self._subjects = value

    @property
    def id_col(self) -> str:
        return self._id_col

    @property
    def dataset_col(self) -> str:
        return self._dataset_col

    @property
    def space(self) -> str | None:
        return self._space

    @property
    def mask_strategy(self) -> str:
        return self._mask_strategy

    @property
    def n_subjects(self) -> int:
        return len(self._subjects)

    def iter_subjects(
        self, order_by: str | None = None
    ) -> Iterator[pd.Series]:  # type: ignore[type-arg]
        """Iterate over subjects one by one.

        Parameters
        ----------
        order_by : str or None
            Column name to sort by before iterating.

        Yields
        ------
        Series
            One row per subject (with the dataset column unwrapped).
        """
        df = self._subjects
        if order_by is not None:
            if order_by not in df.columns:
                raise ConfigError(
                    f"order_by column '{order_by}' not in subjects"
                )
            df = df.sort_values(order_by, na_position="last")

        for _, row in df.iterrows():
            yield row

    def group_map(
        self,
        fn: Callable[..., T],
        *args: Any,
        out: Literal["list", "concat"] = "list",
        order_by: str | None = None,
        on_error: Literal["stop", "warn", "skip"] = "stop",
        **kwargs: Any,
    ) -> list[T] | pd.DataFrame:
        """Apply *fn* to each subject row.

        Parameters
        ----------
        fn : callable
            ``fn(row, *args, **kwargs)`` where *row* is a pandas Series.
        out : str
            ``"list"`` returns results as a list; ``"concat"`` calls
            ``pd.concat`` on DataFrame results.
        order_by : str or None
            Column to sort subjects before iteration.
        on_error : str
            ``"stop"`` (re-raise), ``"warn"`` (warn + skip), ``"skip"``.
        """
        results: list[T] = []
        for row in self.iter_subjects(order_by=order_by):
            try:
                val = fn(row, *args, **kwargs)
            except Exception as exc:
                if on_error == "stop":
                    raise
                if on_error == "warn":
                    warnings.warn(
                        f"group_map: {exc}", stacklevel=2
                    )
                continue
            if val is not None:
                results.append(val)

        if out == "concat" and results:
            return pd.concat(results, ignore_index=True)  # type: ignore[arg-type]
        return results

    def group_reduce(
        self,
        map_fn: Callable[..., U],
        reduce_fn: Callable[[T, U], T],
        init: T,
        *args: Any,
        order_by: str | None = None,
        on_error: Literal["stop", "warn", "skip"] = "stop",
        **kwargs: Any,
    ) -> T:
        """Map then reduce over subjects in a single pass.

        Parameters
        ----------
        map_fn : callable
            Applied to each subject row.
        reduce_fn : callable
            ``reduce_fn(accumulator, mapped_value)`` -> new accumulator.
        init : object
            Initial accumulator value.
        """
        acc: Any = init
        for row in self.iter_subjects(order_by=order_by):
            try:
                val = map_fn(row, *args, **kwargs)
            except Exception as exc:
                if on_error == "stop":
                    raise
                if on_error == "warn":
                    warnings.warn(
                        f"group_reduce: {exc}", stacklevel=2
                    )
                continue
            acc = reduce_fn(acc, val)
        result: T = acc
        return result

    def filter_subjects(
        self, predicate: Callable[[pd.Series], bool]  # type: ignore[type-arg]
    ) -> FmriGroup:
        """Return a new group containing only subjects where *predicate* is True."""
        mask = self._subjects.apply(predicate, axis=1)
        new_df = self._subjects.loc[mask].copy()
        return FmriGroup(
            subjects=new_df,
            id_col=self._id_col,
            dataset_col=self._dataset_col,
            space=self._space,
            mask_strategy=self._mask_strategy,
        )

    def sample_subjects(
        self,
        n: int,
        replace: bool = False,
        strata: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> FmriGroup:
        """Return a new group with *n* randomly sampled subjects.

        Parameters
        ----------
        n : int
            Number of subjects to sample.
        replace : bool
            Sample with replacement.
        strata : str or None
            Column name for stratified sampling.
        rng : Generator or None
            NumPy random generator for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng()

        df = self._subjects

        if strata is None:
            if not replace and n > len(df):
                raise ConfigError(
                    "Cannot sample more subjects than available "
                    "without replacement"
                )
            idx = rng.choice(len(df), size=n, replace=replace)
        else:
            if strata not in df.columns:
                raise ConfigError(
                    f"strata column '{strata}' not in subjects"
                )
            groups = df.groupby(strata)
            idx_parts: list[np.intp] = []
            for _key, group_df in groups:
                group_indices = np.array(group_df.index.tolist())
                pos = np.array(
                    [df.index.get_loc(i) for i in group_indices], dtype=np.intp
                )
                if not replace and n > len(pos):
                    raise ConfigError(
                        f"Cannot sample {n} from stratum with "
                        f"{len(pos)} subjects without replacement"
                    )
                chosen = rng.choice(pos, size=n, replace=replace)
                idx_parts.extend(chosen.tolist())
            idx = np.array(idx_parts, dtype=np.intp)

        new_df = df.iloc[idx].copy().reset_index(drop=True)
        return FmriGroup(
            subjects=new_df,
            id_col=self._id_col,
            dataset_col=self._dataset_col,
            space=self._space,
            mask_strategy=self._mask_strategy,
        )

    def left_join_subjects(
        self,
        other: pd.DataFrame,
        on: str | None = None,
    ) -> FmriGroup:
        """Left-join additional metadata onto the subjects table.

        Parameters
        ----------
        other : DataFrame
            Metadata to join.
        on : str or None
            Join key column. Defaults to the id column.
        """
        key = on or self._id_col
        joined = self._subjects.merge(other, on=key, how="left")
        return FmriGroup(
            subjects=joined,
            id_col=self._id_col,
            dataset_col=self._dataset_col,
            space=self._space,
            mask_strategy=self._mask_strategy,
        )

    def __repr__(self) -> str:
        attrs = [
            c for c in self._subjects.columns if c != self._dataset_col
        ]
        return (
            f"<FmriGroup subjects={self.n_subjects} "
            f"id={self._id_col!r} "
            f"mask_strategy={self._mask_strategy!r} "
            f"attrs={attrs}>"
        )

    def __len__(self) -> int:
        return self.n_subjects


def fmri_group(
    subjects: pd.DataFrame,
    id_col: str,
    dataset_col: str = "dataset",
    space: str | None = None,
    mask_strategy: Literal[
        "subject_specific", "intersect", "union"
    ] = "subject_specific",
) -> FmriGroup:
    """Create an :class:`FmriGroup` from a subjects DataFrame.

    Parameters
    ----------
    subjects : DataFrame
        One row per subject with *id_col* and *dataset_col*.
    id_col : str
        Column with subject identifiers.
    dataset_col : str
        Column storing :class:`FmriDataset` objects.
    space : str or None
        Common space label.
    mask_strategy : str
        ``"subject_specific"``, ``"intersect"``, or ``"union"``.
    """
    return FmriGroup(
        subjects=subjects,
        id_col=id_col,
        dataset_col=dataset_col,
        space=space,
        mask_strategy=mask_strategy,
    )
