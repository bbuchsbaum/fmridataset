# ADR-004: Observation-axis temporal contract

Status: accepted
Date: 2026-08-27

## Context

The canonical frame model has no notion of acquisition timing. An `fmri_frame`
is observations by features; that its observations are often volumes acquired
in runs, at a fixed repetition time, some of them censored, is a fact about
those observations rather than a second array dimension.

`read_bids_bold()` already encoded that fact as ordinary observation metadata —
`run_id`, `TR`, `volume_index`, `run_time` columns, plus a `run` entity keyed on
`scan_id` and an `observation_run` key relation. Nothing validated any of it, so
it was a convention rather than a contract: every consumer that needed run
structure re-derived it, and none could rely on the result.

This blocks real work. `get_TR()`, `blockids()`, `get_run_lengths()`,
`data_chunks(runwise = TRUE)`, `as.matrix_dataset()`, and every legacy dataset
adapter in the 0.11 compatibility shim all need run structure, and there was no
contract for them to be written against.

## Decision

Run structure, repetition time, and censoring are **observation metadata under a
validated schema**, not new frame state.

- `temporal_schema(x)` derives and validates the schema from a frame or view.
- `as_sampling_frame(x)` reconstructs the `fmrihrf::sampling_frame` that the
  design machinery and the legacy accessors consume.
- `has_temporal_schema(x)` reports whether a usable schema is present.

The schema is **derived, never stored**. The columns are the truth, so the schema
cannot go stale, adds nothing to FDS serialization, and follows subsetting,
reordering, and binding for free.

### Columns

| Column | Required | Contract |
|---|---|---|
| run | yes | One value per observation naming its acquisition run. Any type, compared as character. No missing or empty values. |
| `TR` | no | Positive finite seconds, constant within a run. Runs may differ from each other, matching `fmrihrf::sampling_frame()`. |
| `censor` | no | Logical, one per observation, `TRUE` where excluded. No missing values. |

### Which column names the run

The run column is **discovered, not assumed**, in this order:

1. the key of an observation-sourced `key_relation` whose target entity has
   `entity_type == "run"`;
2. `scan_id`;
3. `run_id`;
4. otherwise an error naming what was looked for.

An explicit `run_col` argument overrides all of it.

The obvious choice — `run_id` — is wrong. BIDS `run` is a within-session label,
so a subject with two sessions each containing `run-1` has two distinct
acquisitions sharing one label, and a dataset with no run entity gives every
scan `run-none`. Either case silently merges unrelated acquisitions into a single
run. `scan_id` is unique per acquisition by construction, and a frame that
declares a run entity has already said which column identifies it.

### Order and contiguity

Runs are numbered by **first appearance**, not by sorting, so `block_ids` is
stable under any operation that preserves observation order.

A frame is *contiguous* when each run occupies one unbroken stretch of
observations. Frames are **not required** to be contiguous: `filter_obs()` and
ID-based reordering both produce legal interleaved views, and the schema
describes them correctly. But a `sampling_frame` is a run-length encoding and
cannot represent one, so `as_sampling_frame()` refuses a non-contiguous frame
rather than silently reordering it into a shape the caller did not ask for.

This is the one place the frame model and the legacy `sampling_frame` genuinely
disagree, and the disagreement is made explicit rather than papered over.

## Consequences

The 0.11 compatibility shim can implement `get_TR()`, `blockids()`,
`get_run_lengths()`, `n_timepoints()`, and the runwise branch of `data_chunks()`
as thin readers over `temporal_schema()`, against a contract rather than a
convention.

Frames that carry no run information remain first-class. Beta estimates, parcel
summaries, and latent scores have no acquisition structure, and
`has_temporal_schema()` is `FALSE` for them rather than an error condition.

`volume_index` and `run_time`, which `read_bids_bold()` also writes, are left as
plain annotations. They are derivable from the schema and nothing depends on
them being validated.

Three new exports: `temporal_schema()`, `has_temporal_schema()`,
`as_sampling_frame()`.
