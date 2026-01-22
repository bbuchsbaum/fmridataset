# Get Mask from Backend

Returns a logical mask indicating which voxels contain valid data.

## Usage

``` r
# S3 method for class 'h5_backend'
backend_get_mask(backend)

# S3 method for class 'latent_backend'
backend_get_mask(backend)

# S3 method for class 'matrix_backend'
backend_get_mask(backend)

# S3 method for class 'nifti_backend'
backend_get_mask(backend)

backend_get_mask(backend)

# S3 method for class 'study_backend'
backend_get_mask(backend)

# S3 method for class 'zarr_backend'
backend_get_mask(backend)
```

## Arguments

- backend:

  A storage backend object

## Value

A logical vector satisfying:

- length(mask) == prod(backend_get_dims(backend)\$spatial)

- sum(mask) \> 0 (no empty masks allowed)

- No NA values allowed
