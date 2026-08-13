# Get Spatial Loadings from a Latent-Mode BIDS H5 Dataset

Retrieves the spatial loadings matrix (or matrices) from a latent-mode
BIDS HDF5 archive. Only available when `compression_mode = "latent"`.

## Usage

``` r
get_loadings(x, scan_name = NULL, ...)

# S3 method for class 'bids_h5_study_dataset'
get_loadings(x, scan_name = NULL, ...)
```

## Arguments

- x:

  A `bids_h5_study_dataset` object opened in latent mode.

- scan_name:

  Character. Name of a specific scan, or `NULL` to return loadings for
  all scans as a named list.

- ...:

  Additional arguments passed to methods.

## Value

A numeric matrix `[V, K]` for a single scan, or a named list of such
matrices when `scan_name = NULL`.
