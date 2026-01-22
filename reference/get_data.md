# Get Data from fMRI Dataset Objects

Generic function to extract data from various fMRI dataset types.
Returns the underlying data in its native format (NeuroVec, matrix,
etc.).

## Usage

``` r
get_data(x, ...)
```

## Arguments

- x:

  An fMRI dataset object (e.g., fmri_dataset, matrix_dataset)

- ...:

  Additional arguments passed to methods

## Value

Dataset-specific data object:

- For `fmri_dataset`: Returns the underlying NeuroVec or matrix

- For `matrix_dataset`: Returns the data matrix

## Details

This function extracts the raw data from dataset objects, preserving the
original data type. For NeuroVec-based datasets, returns a NeuroVec
object. For matrix-based datasets, returns a matrix.

## See also

[`get_data_matrix`](https://bbuchsbaum.github.io/fmridataset/reference/get_data_matrix.md)
for extracting data as a matrix,
[`get_mask`](https://bbuchsbaum.github.io/fmridataset/reference/get_mask.md)
for extracting the mask

## Examples

``` r
# \donttest{
# Create a matrix dataset
mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
ds <- matrix_dataset(mat, TR = 2, run_length = 100)

# Extract the data
data <- get_data(ds)
identical(data, mat) # TRUE
#> [1] TRUE
# }
```
