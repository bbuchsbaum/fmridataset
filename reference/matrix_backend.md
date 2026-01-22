# Create a Matrix Backend

Create a Matrix Backend

## Usage

``` r
matrix_backend(data_matrix, mask = NULL, spatial_dims = NULL, metadata = NULL)
```

## Arguments

- data_matrix:

  A matrix in timepoints × voxels orientation

- mask:

  Logical vector indicating which voxels are valid

- spatial_dims:

  Numeric vector of length 3 specifying spatial dimensions

- metadata:

  Optional list of metadata

## Value

A matrix_backend S3 object
