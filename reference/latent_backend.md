# Create a Latent Backend

Creates a storage backend for latent space fMRI data.

## Usage

``` r
latent_backend(source, preload = FALSE)
```

## Arguments

- source:

  Character vector of paths to LatentNeuroVec HDF5 files (.lv.h5) or a
  list of LatentNeuroVec objects from the fmristore or fmrilatent
  packages. When file paths are provided, the fmristore package is
  required for reading. When in-memory LatentNeuroVec objects are
  provided, neither fmristore nor fmrilatent is required at runtime
  (though fmrilatent is used for lazy handle materialization if
  present).

- preload:

  Logical, whether to load all data into memory (default: FALSE)

## Value

A latent_backend S3 object

## Examples

``` r
if (FALSE) { # \dontrun{
# From HDF5 files (requires fmristore)
backend <- latent_backend(c("run1.lv.h5", "run2.lv.h5"))

# From fmristore objects
lvec1 <- fmristore::read_vec("run1.lv.h5")
lvec2 <- fmristore::read_vec("run2.lv.h5")
backend <- latent_backend(list(lvec1, lvec2))

# From fmrilatent objects (no fmristore needed)
lvec <- fmrilatent::encode(data_matrix, spec_time_dct(k = 15), mask = brain_mask)
backend <- latent_backend(list(lvec))
} # }
```
