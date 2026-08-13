# Create an fMRI Dataset from Zarr Arrays

Creates an fMRI dataset object from Zarr array files. Zarr is a
cloud-native array format that supports chunked, compressed storage and
is ideal for large neuroimaging datasets.

## Usage

``` r
fmri_zarr_dataset(
  zarr_source,
  TR,
  run_length,
  event_table = data.frame(),
  censor = NULL,
  preload = FALSE
)
```

## Arguments

- zarr_source:

  Path to Zarr store (directory or URL)

- TR:

  The repetition time in seconds

- run_length:

  Vector of integers indicating the number of scans in each run

- event_table:

  Optional data.frame containing event onsets and experimental variables

- censor:

  Optional binary vector indicating which scans to remove

- preload:

  Whether to load all data into memory (default: FALSE)

## Value

An fMRI dataset object of class c("fmri_file_dataset",
"volumetric_dataset", "fmri_dataset", "list")

## Details

The Zarr backend expects data organized as a 4D array with dimensions
(x, y, z, time). The data is accessed lazily by default, loading only
the requested chunks into memory.

Zarr stores should contain a single 4D array. For mask data, provide it
separately through the fmri_dataset interface if needed.

## Experimental

This function uses the CRAN zarr package which is relatively new
(v0.1.1, Dec 2025). It supports Zarr v3 format only - Zarr v2 stores
cannot be read. Please report any issues to help improve the package.

## See also

[`zarr_backend`](https://bbuchsbaum.github.io/fmridataset/reference/zarr_backend.md),
[`fmri_dataset`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_dataset.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Local Zarr store
dataset <- fmri_zarr_dataset(
  "path/to/data.zarr",
  TR = 2,
  run_length = c(150, 150, 150)
)

# Remote store
dataset <- fmri_zarr_dataset(
  "https://example.com/subject01.zarr",
  TR = 1.5,
  run_length = 300
)

# Preload small dataset into memory
dataset <- fmri_zarr_dataset(
  "small_data.zarr",
  TR = 2,
  run_length = 100,
  preload = TRUE
)
} # }
```
