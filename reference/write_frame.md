# Persist and reopen an fmri frame

These functions provide the semantic-package entry point while
delegating physical HDF5 work to `fmristore`. Reopened assays are
reconstructible lazy sources; opening a frame does not read assay
values.

## Usage

``` r
write_frame(x, path, format = "hdf5", ...)

open_frame(path, format = "hdf5", ...)
```

## Arguments

- x:

  An `fmri_frame`.

- path:

  Destination or source path.

- format:

  Storage format. The walking-skeleton implementation supports `"hdf5"`.

- ...:

  Arguments passed to the physical store implementation.

## Value

`write_frame()` invisibly returns the committed path. `open_frame()`
returns an `fmri_frame`.
