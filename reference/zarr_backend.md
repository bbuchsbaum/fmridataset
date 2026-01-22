# Create a Zarr Backend

Creates a storage backend for Zarr array data.

## Usage

``` r
zarr_backend(
  source,
  data_key = "data",
  mask_key = "mask",
  preload = FALSE,
  cache_size = 100
)
```

## Arguments

- source:

  Character path to Zarr store (directory or zip) or URL for remote
  stores

- data_key:

  Character key for the main data array within the store (default:
  "data")

- mask_key:

  Character key for the mask array (default: "mask"). Set to NULL if no
  mask.

- preload:

  Logical, whether to load all data into memory (default: FALSE)

- cache_size:

  Integer, number of chunks to cache in memory (default: 100)

## Value

A zarr_backend S3 object

## Examples

``` r
if (FALSE) { # \dontrun{
# Local Zarr store
backend <- zarr_backend("path/to/data.zarr")

# Remote S3 store
backend <- zarr_backend("s3://bucket/path/to/data.zarr")

# Custom array keys
backend <- zarr_backend(
  "data.zarr",
  data_key = "fmri/bold",
  mask_key = "fmri/mask"
)
} # }
```
