# Get Data Matrix from fMRI Dataset Objects

Generic function to extract data as a matrix from various fMRI dataset
types. Always returns a matrix with timepoints as rows and voxels as
columns.

## Usage

``` r
get_data_matrix(x, ...)
```

## Arguments

- x:

  An fMRI dataset object (e.g., fmri_dataset, matrix_dataset)

- ...:

  Additional arguments passed to methods

## Value

A numeric matrix with dimensions:

- Rows: Number of timepoints

- Columns: Number of voxels (within mask)

## Details

This function provides a unified interface for accessing fMRI data as a
matrix, regardless of the underlying storage format. The returned matrix
always has timepoints in rows and voxels in columns, matching the
conventional fMRI data organization.

## See also

[`get_data`](https://bbuchsbaum.github.io/fmridataset/reference/get_data.md)
for extracting data in native format,
[`as.matrix_dataset`](https://bbuchsbaum.github.io/fmridataset/reference/as.matrix_dataset.md)
for converting to matrix dataset

## Examples

``` r
# \donttest{
# Create a matrix dataset
mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
ds <- matrix_dataset(mat, TR = 2, run_length = 100)

# Extract as matrix
data_mat <- get_data_matrix(ds)
dim(data_mat) # 100 x 50
#> [1] 100  50
# }
```
