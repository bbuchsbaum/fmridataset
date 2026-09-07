# The observation-axis temporal contract

The canonical frame model carries no acquisition timing of its own: an
`fmri_frame` is observations by features, and when the observations
happen to be volumes acquired in runs, that fact is ordinary observation
metadata. The package already writes it that way –
[`read_bids_bold()`](https://bbuchsbaum.github.io/fmridataset/reference/read_bids_bold.md)
emits `run_id` and `TR` columns – but nothing validated the convention,
so every consumer reinvented it and none could rely on it.

## Usage

``` r
temporal_schema(x, run_col = NULL, tr_col = "TR", censor_col = "censor")

has_temporal_schema(x, ...)

as_sampling_frame(x, ...)
```

## Arguments

- x:

  An `fmri_frame` or `fmri_view`.

- run_col, tr_col, censor_col:

  Observation metadata columns holding the run label, repetition time,
  and censoring indicator.

- ...:

  Passed to `temporal_schema()`.

## Value

`temporal_schema()` returns a `frame_temporal_schema`.
`has_temporal_schema()` returns a scalar logical. `as_sampling_frame()`
returns an
[`fmrihrf::sampling_frame`](https://bbuchsbaum.github.io/fmrihrf/reference/sampling_frame.html).

## Details

These functions make it a contract. `temporal_schema()` derives a
validated description from the observation metadata;
`as_sampling_frame()` reconstructs the
[`fmrihrf::sampling_frame`](https://bbuchsbaum.github.io/fmrihrf/reference/sampling_frame.html)
that the legacy accessors and the design machinery expect.

The schema is derived, never stored. The columns are the truth, so the
schema cannot go stale, needs no serialization of its own, and follows
subsetting, reordering, and binding for free.

## Contract

- `run_id`:

  Required. One value per observation naming the acquisition run it
  belongs to. Any type; compared as character. No missing values.

- `TR`:

  Optional. Repetition time in seconds, positive and finite, and
  constant within each run. Runs may differ from one another, matching
  [`fmrihrf::sampling_frame()`](https://bbuchsbaum.github.io/fmrihrf/reference/sampling_frame.html).

- `censor`:

  Optional. Logical, one value per observation, `TRUE` where the
  observation is to be excluded. No missing values.

## Order and contiguity

Runs are numbered in order of first appearance, not by sorting, so
`block_ids` is stable under any operation that preserves observation
order. A frame is *contiguous* when each run occupies one unbroken
stretch of observations. Frames are not required to be contiguous –
[`filter_obs()`](https://bbuchsbaum.github.io/fmridataset/reference/filter_obs.md)
and ID-based reordering both produce legal interleaved views – but a
`sampling_frame` is a run-length encoding and cannot represent one, so
`as_sampling_frame()` refuses a non-contiguous frame rather than
silently reordering it.

## Examples

``` r
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(12), 6, 2)),
  observations = data.frame(
    .obs_id = sprintf("t%02d", 1:6),
    run_id = rep(c("run-1", "run-2"), each = 3),
    TR = 2
  )
)
schema <- temporal_schema(frame)
schema$run_lengths
#> run-1 run-2 
#>     3     3 
as_sampling_frame(frame)
#> Sampling Frame
#> ==============
#> 
#> Structure:
#>   2 blocks
#>   Total scans: 6
#> 
#> Timing:
#>   TR: 2 s
#>   Precision: 0.1 s
#> 
#> Duration:
#>   Total time: 12.0 s
```
