# Convert fmri_series to Matrix

This method realizes the underlying lazy matrix and returns an ordinary
matrix with timepoints in rows and voxels in columns.

## Usage

``` r
# S3 method for class 'fmri_series'
as.matrix(x, ...)
```

## Arguments

- x:

  An `fmri_series` object

- ...:

  Additional arguments (ignored)

## Value

A matrix with timepoints as rows and voxels as columns

## See also

[`fmri_series`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_series.md)
for the class definition,
[`as_tibble.fmri_series`](https://bbuchsbaum.github.io/fmridataset/reference/as_tibble.fmri_series.md)
for tibble conversion

## Examples

``` r
# \donttest{
# Create small example
mat <- matrix(rnorm(20), nrow = 4, ncol = 5)

backend <- matrix_backend(mat, mask = rep(TRUE, ncol(mat)))
dataset <- fmri_dataset(backend, TR = 1, run_length = nrow(mat))
fs <- fmri_series(dataset)

# Convert to matrix
mat_result <- as.matrix(fs)
dim(mat_result) # 4 x 5
#> [1] 4 5
# }
```
