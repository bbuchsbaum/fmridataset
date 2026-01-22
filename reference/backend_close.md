# Close a Storage Backend

Closes a storage backend and releases any resources. Stateless backends
can implement this as a no-op.

## Usage

``` r
# S3 method for class 'h5_backend'
backend_close(backend)

# S3 method for class 'latent_backend'
backend_close(backend)

# S3 method for class 'matrix_backend'
backend_close(backend)

# S3 method for class 'nifti_backend'
backend_close(backend)

backend_close(backend)

# S3 method for class 'study_backend'
backend_close(backend)

# S3 method for class 'zarr_backend'
backend_close(backend)
```

## Arguments

- backend:

  A storage backend object

## Value

NULL (invisibly)
