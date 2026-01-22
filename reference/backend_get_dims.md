# Get Dimensions from Backend

Returns the dimensions of the data stored in the backend.

## Usage

``` r
# S3 method for class 'h5_backend'
backend_get_dims(backend)

# S3 method for class 'latent_backend'
backend_get_dims(backend)

# S3 method for class 'matrix_backend'
backend_get_dims(backend)

# S3 method for class 'nifti_backend'
backend_get_dims(backend)

backend_get_dims(backend)

# S3 method for class 'study_backend'
backend_get_dims(backend)

# S3 method for class 'zarr_backend'
backend_get_dims(backend)
```

## Arguments

- backend:

  A storage backend object

## Value

A named list with elements:

- spatial: numeric vector of length 3 (x, y, z dimensions)

- time: integer, number of timepoints
