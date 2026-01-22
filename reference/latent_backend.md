# Create a Latent Backend

Creates a storage backend for latent space fMRI data.

## Usage

``` r
latent_backend(source, preload = FALSE)
```

## Arguments

- source:

  Character vector of paths to LatentNeuroVec HDF5 files (.lv.h5) or a
  list of LatentNeuroVec objects from the fmristore package.

- preload:

  Logical, whether to load all data into memory (default: FALSE)

## Value

A latent_backend S3 object

## Examples

``` r
if (FALSE) { # \dontrun{
# From HDF5 files
backend <- latent_backend(c("run1.lv.h5", "run2.lv.h5"))

# From pre-loaded objects
lvec1 <- fmristore::read_vec("run1.lv.h5")
lvec2 <- fmristore::read_vec("run2.lv.h5")
backend <- latent_backend(list(lvec1, lvec2))
} # }
```
