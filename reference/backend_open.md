# Open a Storage Backend

Opens a storage backend and acquires any necessary resources (e.g., file
handles). Stateless backends can implement this as a no-op.

## Usage

``` r
# S3 method for class 'h5_backend'
backend_open(backend)

# S3 method for class 'latent_backend'
backend_open(backend)

# S3 method for class 'matrix_backend'
backend_open(backend)

# S3 method for class 'nifti_backend'
backend_open(backend)

backend_open(backend)

# S3 method for class 'study_backend'
backend_open(backend)

# S3 method for class 'zarr_backend'
backend_open(backend)
```

## Arguments

- backend:

  A storage backend object

## Value

The backend object (possibly modified with state)
