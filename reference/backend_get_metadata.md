# Get Metadata from Backend

Returns metadata associated with the data (e.g., affine matrix, voxel
dimensions).

## Usage

``` r
# S3 method for class 'h5_backend'
backend_get_metadata(backend)

# S3 method for class 'latent_backend'
backend_get_metadata(backend)

# S3 method for class 'matrix_backend'
backend_get_metadata(backend)

# S3 method for class 'nifti_backend'
backend_get_metadata(backend)

backend_get_metadata(backend)

# S3 method for class 'zarr_backend'
backend_get_metadata(backend)
```

## Arguments

- backend:

  A storage backend object

## Value

A list containing neuroimaging metadata, which may include:

- affine: 4x4 affine transformation matrix

- voxel_dims: numeric vector of voxel dimensions

- intent_code: NIfTI intent code

- Additional format-specific metadata
