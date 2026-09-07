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

`write_frame()` invisibly returns the committed path, normalized with
forward slashes on every platform. `open_frame()` returns an
`fmri_frame`.

## Details

Neither function computes a content hash. Persistence records semantic
manifest digests and source fingerprints only; a caller who wants a
value receipt for the written or reopened arrays requests it explicitly
with
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
and records the result where it is needed.

## Examples

``` r
if (requireNamespace("fmristore", quietly = TRUE)) {
  src <- memory_source(matrix(seq_len(6), nrow = 2))
  obs <- tibble::tibble(.obs_id = c("o1", "o2"))
  space <- index_space(3, id_policy = "deterministic", namespace = "demo")
  frame <- fmri_frame(list(beta = src), obs, space = space)
  path <- tempfile(fileext = ".h5")
  committed <- write_frame(frame, path)
  reopened <- open_frame(committed)
  collect_assay(reopened, "beta")
}
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
#> [2,]    2    4    6
```
