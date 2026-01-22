# Get Mask from fMRI Dataset Objects

Generic function to extract masks from various fMRI dataset types.
Returns the mask in its appropriate format for the dataset type.

## Usage

``` r
get_mask(x, ...)
```

## Arguments

- x:

  An fMRI dataset object (e.g., fmri_dataset, matrix_dataset)

- ...:

  Additional arguments passed to methods

## Value

Mask object appropriate for the dataset type:

- For `matrix_dataset`: Logical vector

- For `fmri_dataset`: NeuroVol or logical vector

## Details

The mask defines which voxels are included in the analysis. Different
dataset types may store masks in different formats (logical vectors,
NeuroVol objects, etc.). This function provides a unified interface for
mask extraction.

## See also

[`get_data`](https://bbuchsbaum.github.io/fmridataset/reference/get_data.md)
for extracting data,
[`get_data_matrix`](https://bbuchsbaum.github.io/fmridataset/reference/get_data_matrix.md)
for extracting data as matrix

## Examples

``` r
# \donttest{
# Create a matrix dataset (matrix_dataset creates default mask internally)
mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
ds <- matrix_dataset(mat, TR = 2, run_length = 100)

# Extract the mask (matrix_dataset creates all TRUE mask by default)
extracted_mask <- get_mask(ds)
sum(extracted_mask) # 50 (all TRUE values)
#> [1] 50
# }
```
