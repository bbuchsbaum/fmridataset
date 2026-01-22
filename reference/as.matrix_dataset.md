# Convert to Matrix Dataset

Generic function to convert various fMRI dataset types to matrix_dataset
objects. Provides a unified interface for getting matrix-based
representations.

## Usage

``` r
as.matrix_dataset(x, ...)
```

## Arguments

- x:

  An fMRI dataset object

- ...:

  Additional arguments passed to methods

## Value

A matrix_dataset object with the same data as the input

## Details

This function converts different dataset representations to the standard
matrix_dataset format, which stores data as a matrix with timepoints in
rows and voxels in columns. This is useful for algorithms that require
matrix operations or when a consistent data format is needed.

## See also

[`matrix_dataset`](https://bbuchsbaum.github.io/fmridataset/reference/matrix_dataset.md)
for creating matrix datasets,
[`get_data_matrix`](https://bbuchsbaum.github.io/fmridataset/reference/get_data_matrix.md)
for extracting data as matrix

## Examples

``` r
# \donttest{
# Convert various dataset types to matrix_dataset
# (example requires actual dataset object)
# mat_ds <- as.matrix_dataset(some_dataset)
# }
```
