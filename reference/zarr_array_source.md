# Construct an experimental Zarr array source

`zarr_array_source()` describes one two-dimensional Zarr array whose
logical axes are observations and features. The descriptor contains no
open store, R6 object, external pointer, or loader function. Runtime
handles are opened only for metadata discovery and numerical reads.

## Usage

``` r
zarr_array_source(
  uri,
  array_path = "/",
  shape = NULL,
  dtype = NULL,
  chunks = NULL,
  physical_axes = c("observation", "feature")
)
```

## Arguments

- uri:

  One local path, file URI, or HTTP(S) location understood by
  [`zarr::open_zarr()`](https://r-cf.github.io/zarr/reference/open_zarr.html).

- array_path:

  Absolute path of the array within the Zarr hierarchy. The default
  `"/"` denotes a single-array store.

- shape:

  Optional logical observation-by-feature shape.

- dtype:

  Optional logical storage dtype supported by `ArraySource`.

- chunks:

  Optional logical observation-by-feature chunk shape.

- physical_axes:

  Names of the two physical Zarr dimensions, permitting either
  observation-first or feature-first storage.

## Value

A serializable `zarr_array_source` descriptor.

## Details

The optional `zarr` package is needed only when metadata must be
discovered or data are read. Supplying `shape`, `dtype`, and `chunks`
together therefore permits metadata-only construction and serialization
on workers where Zarr is not installed.
