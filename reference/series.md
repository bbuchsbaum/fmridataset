# Deprecated alias for `fmri_series`

`series()` forwards to
[`fmri_series()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_series.md)
for backward compatibility. A deprecation warning is emitted once per
session.

## Usage

``` r
series(
  dataset,
  selector = NULL,
  timepoints = NULL,
  output = c("fmri_series", "DelayedMatrix"),
  event_window = NULL,
  ...
)
```

## Arguments

- dataset:

  An `fmri_dataset` object.

- selector:

  Spatial selector or `NULL` for all voxels.

- timepoints:

  Optional temporal subset or `NULL` for all.

- output:

  Return type - "FmriSeries" (default) or "DelayedMatrix".

- event_window:

  Reserved for future use.

- ...:

  Additional arguments passed to methods.

## Value

See
[`fmri_series()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_series.md)
