"""Tests for FmriGroup and group operations."""

import numpy as np
import pandas as pd
import pytest

from fmridataset import matrix_dataset
from fmridataset.errors import ConfigError
from fmridataset.fmri_group import FmriGroup, fmri_group


@pytest.fixture()
def three_subject_group():
    rng = np.random.default_rng(42)
    datasets = []
    for _ in range(3):
        mat = rng.standard_normal((10, 5)).astype(np.float64)
        datasets.append(matrix_dataset(mat, TR=2.0, run_length=10))

    df = pd.DataFrame(
        {
            "subject_id": ["S01", "S02", "S03"],
            "age": [25, 30, 35],
            "dataset": datasets,
        }
    )
    return fmri_group(df, id_col="subject_id")


class TestFmriGroupConstruction:
    def test_basic(self, three_subject_group) -> None:
        grp = three_subject_group
        assert grp.n_subjects == 3
        assert len(grp) == 3
        assert grp.id_col == "subject_id"
        assert grp.dataset_col == "dataset"
        assert grp.mask_strategy == "subject_specific"

    def test_missing_id_col(self) -> None:
        df = pd.DataFrame({"x": [1], "dataset": [None]})
        with pytest.raises(ConfigError, match="id column"):
            fmri_group(df, id_col="subject_id")

    def test_missing_dataset_col(self) -> None:
        df = pd.DataFrame({"subject_id": ["S01"]})
        with pytest.raises(ConfigError, match="dataset column"):
            fmri_group(df, id_col="subject_id")

    def test_repr(self, three_subject_group) -> None:
        r = repr(three_subject_group)
        assert "FmriGroup" in r
        assert "subjects=3" in r

    def test_space(self) -> None:
        df = pd.DataFrame({"sid": ["A"], "dataset": [None]})
        grp = fmri_group(df, id_col="sid", space="MNI152")
        assert grp.space == "MNI152"


class TestIterSubjects:
    def test_iter_all(self, three_subject_group) -> None:
        rows = list(three_subject_group.iter_subjects())
        assert len(rows) == 3

    def test_iter_order_by(self, three_subject_group) -> None:
        rows = list(three_subject_group.iter_subjects(order_by="age"))
        ages = [r["age"] for r in rows]
        assert ages == sorted(ages)

    def test_iter_bad_column(self, three_subject_group) -> None:
        with pytest.raises(ConfigError, match="order_by"):
            list(three_subject_group.iter_subjects(order_by="nonexistent"))


class TestGroupMap:
    def test_map_list(self, three_subject_group) -> None:
        result = three_subject_group.group_map(lambda row: row["age"] * 2)
        assert result == [50, 60, 70]

    def test_map_on_error_skip(self, three_subject_group) -> None:
        def fail_on_s02(row):
            if row["subject_id"] == "S02":
                raise ValueError("boom")
            return row["age"]

        result = three_subject_group.group_map(
            fail_on_s02, on_error="skip"
        )
        assert result == [25, 35]

    def test_map_on_error_stop(self, three_subject_group) -> None:
        def always_fail(row):
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            three_subject_group.group_map(always_fail, on_error="stop")

    def test_map_on_error_warn(self, three_subject_group) -> None:
        def fail_on_s01(row):
            if row["subject_id"] == "S01":
                raise ValueError("bad")
            return row["age"]

        with pytest.warns(UserWarning, match="bad"):
            result = three_subject_group.group_map(
                fail_on_s01, on_error="warn"
            )
        assert result == [30, 35]


class TestGroupReduce:
    def test_sum_ages(self, three_subject_group) -> None:
        total = three_subject_group.group_reduce(
            map_fn=lambda row: row["age"],
            reduce_fn=lambda acc, v: acc + v,
            init=0,
        )
        assert total == 90

    def test_reduce_skip_error(self, three_subject_group) -> None:
        def maybe_fail(row):
            if row["subject_id"] == "S02":
                raise ValueError("skip me")
            return row["age"]

        total = three_subject_group.group_reduce(
            map_fn=maybe_fail,
            reduce_fn=lambda acc, v: acc + v,
            init=0,
            on_error="skip",
        )
        assert total == 60  # S01(25) + S03(35)


class TestFilterSubjects:
    def test_filter(self, three_subject_group) -> None:
        grp2 = three_subject_group.filter_subjects(lambda r: r["age"] >= 30)
        assert grp2.n_subjects == 2

    def test_filter_none(self, three_subject_group) -> None:
        grp2 = three_subject_group.filter_subjects(lambda r: r["age"] > 100)
        assert grp2.n_subjects == 0


class TestSampleSubjects:
    def test_sample(self, three_subject_group) -> None:
        grp2 = three_subject_group.sample_subjects(
            2, rng=np.random.default_rng(0)
        )
        assert grp2.n_subjects == 2

    def test_sample_with_replacement(self, three_subject_group) -> None:
        grp2 = three_subject_group.sample_subjects(
            5, replace=True, rng=np.random.default_rng(0)
        )
        assert grp2.n_subjects == 5

    def test_sample_too_many(self, three_subject_group) -> None:
        with pytest.raises(ConfigError, match="Cannot sample"):
            three_subject_group.sample_subjects(10, replace=False)


class TestLeftJoinSubjects:
    def test_join(self, three_subject_group) -> None:
        meta = pd.DataFrame(
            {"subject_id": ["S01", "S02", "S03"], "group": ["A", "B", "A"]}
        )
        grp2 = three_subject_group.left_join_subjects(meta)
        assert "group" in grp2.subjects.columns
        assert grp2.subjects["group"].tolist() == ["A", "B", "A"]
