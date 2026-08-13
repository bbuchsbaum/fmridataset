# Get Data from Backend

Reads data from the backend in canonical timepoints × voxels
orientation.

## Usage

``` r
# S3 method for class 'bids_h5_scan_backend'
backend_get_data(backend, rows = NULL, cols = NULL)

# S3 method for class 'h5_backend'
backend_get_data(backend, rows = NULL, cols = NULL)

# S3 method for class 'latent_backend'
backend_get_data(backend, rows = NULL, cols = NULL)

# S3 method for class 'matrix_backend'
backend_get_data(backend, rows = NULL, cols = NULL)

# S3 method for class 'nifti_backend'
backend_get_data(backend, rows = NULL, cols = NULL)

backend_get_data(backend, rows = NULL, cols = NULL)

# S3 method for class 'study_backend'
backend_get_data(backend, rows = NULL, cols = NULL)

# S3 method for class 'zarr_backend'
backend_get_data(backend, rows = NULL, cols = NULL)
```

## Arguments

- backend:

  A storage backend object

- rows:

  Integer vector of row indices (timepoints) to read, or NULL for all

- cols:

  Integer vector of column indices (voxels) to read, or NULL for all

## Value

A matrix in timepoints × voxels orientation

## Details

For `compression_mode = "parcellated"` reads
`/scans/<name>/data/summary_data` (`[T, K]`). For
`compression_mode = "latent"` reads `/scans/<name>/data/basis`
(`[T, K]`). In both cases the return value is a `[T, K]` numeric matrix.
