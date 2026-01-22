# fmri_series: fMRI Time Series Container

An S3 class representing lazily accessed fMRI time series data. The
underlying data is stored in a lazy matrix (typically a `delarr` object)
with rows corresponding to timepoints and columns corresponding to
voxels.

Core interface for retrieving voxel time series from fMRI datasets.

## Usage

``` r
fmri_series(
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

An object of class `fmri_series`

An `fmri_series` (with a `delarr` lazy matrix payload) or a
`DelayedMatrix` when `output = "DelayedMatrix"`.

## Details

An fmri_series object contains:

- `data`: A lazy matrix with timepoints as rows and voxels as columns

- `voxel_info`: A data.frame containing spatial metadata for each voxel

- `temporal_info`: A data.frame containing metadata for each timepoint

- `selection_info`: A list describing how the data were selected

- `dataset_info`: A list describing the source dataset and backend

## See also

[`as.matrix.fmri_series`](https://bbuchsbaum.github.io/fmridataset/reference/as.matrix.fmri_series.md)
for converting to standard matrix,
[`as_tibble.fmri_series`](https://bbuchsbaum.github.io/fmridataset/reference/as_tibble.fmri_series.md)
for converting to tibble format

## Examples

``` r
# \donttest{
# Create example fmri_series object
mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
backend <- matrix_backend(mat, mask = rep(TRUE, ncol(mat)))
dataset <- fmri_dataset(backend, TR = 1, run_length = rep(25, 4))
fs <- fmri_series(dataset)
fs
#> <fmri_series> 50 voxels x 100 timepoints (lazy)
#> Selector: NULL | Backend: matrix_backend | Orientation: time x voxels
# }
```
