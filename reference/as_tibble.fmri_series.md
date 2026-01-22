# Convert fmri_series to Tibble

The returned tibble contains one row per voxel/timepoint combination
with metadata columns from `temporal_info` and `voxel_info` and a
`signal` column with the data values.

## Usage

``` r
# S3 method for class 'fmri_series'
as_tibble(x, ...)
```

## Arguments

- x:

  An `fmri_series` object

- ...:

  Additional arguments (ignored)

## Value

A tibble with columns from temporal_info, voxel_info, and a signal
column containing the fMRI signal values

## See also

[`fmri_series`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_series.md)
for the class definition,
[`as.matrix.fmri_series`](https://bbuchsbaum.github.io/fmridataset/reference/as.matrix.fmri_series.md)
for matrix conversion

## Examples

``` r
# \donttest{
# Create small example
mat <- matrix(rnorm(12), nrow = 3, ncol = 4)

backend <- matrix_backend(mat, mask = rep(TRUE, ncol(mat)))
dataset <- fmri_dataset(backend, TR = 1, run_length = nrow(mat))
fs <- fmri_series(dataset)

# Convert to tibble
tbl_result <- tibble::as_tibble(fs)
# Result has 12 rows (3 timepoints x 4 voxels)
# with columns: time, voxel, signal
# }
```
