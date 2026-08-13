# Create a Zarr Backend

Creates a storage backend for Zarr array data using the CRAN zarr
package.

## Usage

``` r
zarr_backend(source, preload = FALSE)
```

## Arguments

- source:

  Character path to Zarr store (directory or URL for remote stores)

- preload:

  Logical, whether to load all data into memory (default: FALSE)

## Value

A zarr_backend S3 object

## Experimental

This backend uses the CRAN zarr package which is relatively new (v0.1.1,
Dec 2025). It supports Zarr v3 format only - Zarr v2 stores cannot be
read. Please report any issues to help improve the package.

## Examples

``` r
if (FALSE) { # \dontrun{
# Local Zarr store
backend <- zarr_backend("path/to/data.zarr")

# Remote store
backend <- zarr_backend("https://example.com/data.zarr")
} # }
```
