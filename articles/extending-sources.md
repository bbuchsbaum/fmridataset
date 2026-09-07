# Implementing an array source

Every assay in a frame is backed by an *array source*. A source is a
serializable description of a two-dimensional numerical array plus the
S3 methods that read from it. The package ships in-memory, NIfTI, HDF5
(through `fmristore`), and experimental Zarr sources. A storage package
adds its own by implementing a small set of generics. It then passes the
same conformance checks the built-in sources pass.

This vignette implements a complete source backed by an `.rds` file, in
about sixty lines, and validates it. The file format is deliberately
trivial so the protocol is the whole lesson.

``` r

library(fmridataset)
```

## The protocol

An array source is an S3 object with class
`c("<your_class>", "array_source")` and methods for these generics:

| Generic | Returns |
|----|----|
| [`source_shape()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | Two non-negative integers: observations, features |
| [`source_dtype()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | One storage dtype such as `"float64"`, `"float32"`, `"int16"`, `"logical"` |
| [`source_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | Two positive integers bounded by the shape: the natural read grid |
| [`source_capabilities()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | Unique strings, including `"block_slice"` and `"serializable"` |
| [`source_fingerprint()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | One non-empty string identifying this descriptor and revision |
| [`source_open()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | A handle inheriting from `"array_source_handle"` |
| [`source_read()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | An observation-by-feature matrix for integer selectors, in requested order |
| [`source_read_native()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | A native spatial object, or an error if the source has none |
| [`source_close()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md) | Releases the handle |

The package provides
[`source_descriptor()`](https://bbuchsbaum.github.io/fmridataset/reference/source_descriptor.md),
[`validate_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/source_descriptor.md),
[`source_realization_cost()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md),
and
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
on top of these; you do not implement them.

Two rules hold everything together.

**A descriptor is not a handle.** The source object holds only plain
serializable values: paths, shapes, dtypes, sizes, timestamps, tokens.
It holds no open connections, environments, external pointers, or
closures.
[`validate_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/source_descriptor.md)
rejects a descriptor that contains any of them. Runtime state belongs in
the handle that
[`source_open()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
returns. It never becomes semantic state.

**Fingerprints are cheap and content hashes are explicit.** The
fingerprint is computed once, at construction, and stored in the
descriptor. Its inputs are the descriptor fields and any physical
evidence you can observe without reading values. It must never hash the
array. Value identity is
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md),
which the package computes by streaming your
[`source_read()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md).

## An RDS-backed source

The constructor reads the file once to learn the shape, keeps the file’s
size and modification time as revision evidence, computes the
fingerprint, and validates itself.
[`canonical_sha256()`](https://bbuchsbaum.github.io/fmridataset/reference/canonical-encoding.md)
is the package’s canonical hash and is the right tool for a fingerprint.

``` r

rds_source <- function(path, chunks = NULL) {
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  values <- readRDS(path)
  if (!is.matrix(values) || !is.numeric(values)) {
    stop("rds_source() requires a numeric matrix.", call. = FALSE)
  }
  shape <- as.integer(dim(values))
  if (is.null(chunks)) chunks <- pmax(1L, shape)
  info <- file.info(path)
  out <- structure(
    list(
      path = path,
      shape = shape,
      dtype = "float64",
      chunks = pmin(as.integer(chunks), pmax(1L, shape)),
      size = as.numeric(info$size),
      mtime = as.numeric(info$mtime),
      schema_version = 1L
    ),
    class = c("rds_source", "array_source")
  )
  out$fingerprint <- canonical_sha256(list(
    type = "rds_source",
    schema_version = out$schema_version,
    path = out$path,
    shape = out$shape,
    dtype = out$dtype,
    chunks = out$chunks,
    size = out$size,
    mtime = out$mtime
  ))
  validate_array_source(out)
}
```

The descriptor accessors return stored fields.

``` r

source_shape.rds_source <- function(x, ...) x$shape
source_dtype.rds_source <- function(x, ...) x$dtype
source_chunks.rds_source <- function(x, ...) x$chunks
source_capabilities.rds_source <- function(x, ...) {
  c("row_slice", "column_slice", "block_slice", "serializable")
}
source_fingerprint.rds_source <- function(x, ...) x$fingerprint
```

Opening re-observes the file and compares it with the evidence captured
at construction. When they differ the descriptor is stale. That is not
an I/O failure: the file opened fine, it is simply not the file the
descriptor was made from. The package reserves
`fmridataset_error_source_stale` for this case and
`fmridataset_error_backend_io` for genuine failures. Callers rely on the
distinction. Extension packages raise both as plain conditions with
those classes.

``` r

rds_assert_fresh <- function(x) {
  info <- file.info(x$path)
  actual <- list(size = as.numeric(info$size), mtime = as.numeric(info$mtime))
  expected <- list(size = x$size, mtime = x$mtime)
  if (!identical(actual, expected)) {
    stop(structure(
      list(
        message = paste("RDS file changed since the descriptor was built:", x$path),
        call = NULL,
        source = list(type = "rds_source", path = x$path),
        expected = expected,
        actual = actual
      ),
      class = c("fmridataset_error_source_stale", "fmridataset_error", "error", "condition")
    ))
  }
  invisible(TRUE)
}

source_open.rds_source <- function(x, ...) {
  rds_assert_fresh(x)
  structure(
    list(source = x, values = readRDS(x$path)),
    class = c("rds_source_handle", "array_source_handle")
  )
}
source_close.rds_source <- function(x, ...) invisible(TRUE)
```

Reads take integer selectors on both axes and honor the requested order,
and return a zero-extent matrix for an empty selection. The selection
law (`inst/architecture/ADR-010-selection-algebra.md`) is checked at the
[`source_read()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
generic before your method runs: positions arrive whole, in bounds, and
never repeated, so a method need not defend against a repeated or
out-of-range position, though refusing one again is harmless. Reading
through the handle is the real implementation; reading from the
descriptor opens, reads, and closes.

``` r

source_read.rds_source_handle <- function(x, observations = NULL, features = NULL, ...) {
  values <- x$values
  if (is.null(observations)) observations <- seq_len(nrow(values))
  if (is.null(features)) features <- seq_len(ncol(values))
  if (any(observations < 1L | observations > nrow(values)) ||
      any(features < 1L | features > ncol(values))) {
    stop("rds_source selector is out of bounds.", call. = FALSE)
  }
  values[observations, features, drop = FALSE]
}

source_read.rds_source <- function(x, observations = NULL, features = NULL, ...) {
  handle <- source_open(x)
  on.exit(source_close(handle), add = TRUE)
  source_read(handle, observations = observations, features = features, ...)
}

source_read_native.rds_source <- function(x, observations = NULL, ...) {
  stop("rds_source has no native spatial representation.", call. = FALSE)
}
```

That is the whole source. In a package these methods would be registered
in the `NAMESPACE` with `S3method()`; in a vignette, defining them at
top level is enough for dispatch.

## Try it

``` r

set.seed(4)
reference <- matrix(round(rnorm(30), 2), nrow = 6, ncol = 5)
path <- file.path(tempdir(), "toy.rds")
saveRDS(reference, path)

src <- rds_source(path, chunks = c(2L, 5L))
source_descriptor(src)
#> $shape
#> [1] 6 5
#> 
#> $dtype
#> [1] "float64"
#> 
#> $chunks
#> [1] 2 5
#> 
#> $capabilities
#> [1] "row_slice"    "column_slice" "block_slice"  "serializable"
#> 
#> $fingerprint
#> [1] "9b1777a7987808c3dc821fa2633e6c95c2d9a599c9d3ef602a889a7447f99bea"
```

Reads honor order and shape.

``` r

source_read(src, observations = c(6L, 1L), features = c(5L, 2L))
#>      [,1]  [,2]
#> [1,] 1.24  0.02
#> [2,] 0.59 -1.28
reference[c(6, 1), c(5, 2)]
#>      [,1]  [,2]
#> [1,] 1.24  0.02
#> [2,] 0.59 -1.28
dim(source_read(src, observations = integer(), features = 1:2))
#> [1] 0 2
```

The source drops straight into a frame.
[`explain()`](https://bbuchsbaum.github.io/fmridataset/reference/explain.md)
reports it by class, and views over it are as lazy as any other.

``` r

frame <- fmri_frame(
  assays = list(bold = src),
  observations = data.frame(.obs_id = sprintf("t%02d", 1:6)),
  space = index_space(5L, ids = paste0("f", 1:5))
)
explain(frame)$assays$bold[c("source_type", "chunks")]
#> $source_type
#> [1] "rds_source"
#> 
#> $chunks
#> [1] 2 5
collect_assay(frame[c("t02", "t01"), "f5"])
#>       [,1]
#> [1,] -0.28
#> [2,]  0.59
```

The cost estimate the budget checks use comes for free from the shape
and dtype.

``` r

source_realization_cost(src, observations = 1:2)[c("storage_bytes", "estimated_peak_bytes")]
#> $storage_bytes
#> [1] 80
#> 
#> $estimated_peak_bytes
#> [1] 160
```

## Validate it

[`validate_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/source_descriptor.md)
checks the descriptor contract: two-dimensional shape, supported dtype,
chunk grid bounded by the shape, the required capabilities, a non-empty
fingerprint, and no runtime state.

``` r

validate_array_source(src)
```

A descriptor that smuggles in a function, however well-formed otherwise,
fails.

``` r

leaky <- src
leaky$loader <- function() readRDS(leaky$path)
validate_array_source(leaky)
#> Error:
#> ! Canonical source descriptors cannot contain functions, environments, or external pointers.
```

The behavioral laws live in the package’s conformance helper,
`tests/testthat/helper-frame-conformance.R`. It is a function that takes
a source and its reference matrix. A storage package should copy that
helper into its own tests and call it on every source type it ships. It
checks, in order:

- the descriptor is valid and its shape matches the reference;
- the descriptor survives
  [`serialize()`](https://rdrr.io/r/base/serialize.html) and
  [`unserialize()`](https://rdrr.io/r/base/serialize.html) with the same
  fingerprint, shape, dtype, and chunks;
- opening and closing a handle is safe and leaves the fingerprint
  untouched;
- reads match the reference for full, reversed, permuted, and empty
  selections, and are pure;
- out-of-range and repeated selections raise an error, and views
  composed over the source read what the equivalent flat positions read;
- [`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
  equals the content hash of a memory source over the reference, is
  accepted by
  [`identity_descriptor()`](https://bbuchsbaum.github.io/fmridataset/reference/identity_descriptor.md)
  as a content receipt, and does not change the fingerprint.

The last law is the one that matters most for interoperability, and the
cheapest to check by hand.

``` r

restored <- unserialize(serialize(src, NULL))
identical(source_fingerprint(restored), source_fingerprint(src))
#> [1] TRUE

handle <- source_open(src)
class(handle)
#> [1] "rds_source_handle"   "array_source_handle"
source_close(handle)

identical(content_hash(src), content_hash(memory_source(reference)))
#> [1] TRUE
```

[`counting_source()`](https://bbuchsbaum.github.io/fmridataset/reference/counting_source.md)
wraps any source and records what reaches it. That is how the zero-I/O
laws are certified: build a frame over the wrapper, take views, and
assert the read count is still zero.

``` r

counted <- counting_source(src)
view <- source_view(counted, observations = 1:2)
source_counts(counted)$reads
#> [1] 0
invisible(source_read(view))
source_counts(counted)[c("reads", "values")]
#> $reads
#> [1] 1
#> 
#> $values
#> [1] 10
```

Staleness behaves as designed. Rewriting the file does not change the
fingerprint stored in the descriptor. Every read now fails with the
stale class, and a fresh descriptor has a new fingerprint.

``` r

Sys.sleep(1.1)
saveRDS(reference * 2, path)
class(tryCatch(source_read(src, 1L, 1L), error = identity))[1]
#> [1] "fmridataset_error_source_stale"
identical(source_fingerprint(rds_source(path)), source_fingerprint(src))
#> [1] FALSE
```

## Composition comes for free

The protocol is uniform, so the package’s compositions accept the new
source without knowing anything about it.
[`row_sharded_source()`](https://bbuchsbaum.github.io/fmridataset/reference/row_sharded_source.md)
presents several sources as one array with stable shard IDs and an
inspectable manifest. That is how multi-run and multi-subject data are
bound.

``` r

saveRDS(reference[1:2, ], run_1 <- file.path(tempdir(), "run-1.rds"))
saveRDS(reference[3:6, ], run_2 <- file.path(tempdir(), "run-2.rds"))
sharded <- row_sharded_source(
  list(rds_source(run_1), rds_source(run_2)),
  shard_ids = c("run-1", "run-2")
)
shard_manifest(sharded)[, c(".shard_id", ".start", ".end")]
#>   .shard_id .start .end
#> 1     run-1      1    2
#> 2     run-2      3    6
locate_source_rows(sharded, c(5L, 1L))[, c(".observation", ".shard_id", ".local_observation")]
#>   .observation .shard_id .local_observation
#> 1            5     run-2                  3
#> 2            1     run-1                  1
identical(content_hash(sharded), content_hash(memory_source(reference)))
#> [1] TRUE
```

[`as_delarr()`](https://bbuchsbaum.github.io/fmridataset/reference/as_delarr.md)
wraps any source as a lazy `delarr` array, so chunk-aware numerical
plans can pull from it through
[`source_read()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
under the same budget.

``` r

lazy <- as_delarr(sharded)
dim(lazy)
#> [1] 6 5
```

## Checklist for a storage package

Keep the descriptor serializable and put every handle behind
[`source_open()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md).
Compute the fingerprint at construction from descriptor fields and cheap
physical evidence, never from values. Raise
`fmridataset_error_source_stale` when the evidence no longer matches.
Raise `fmridataset_error_backend_io` when the backend fails. Return
blocks in the requested order, non-dropping, and refuse out-of-range
selectors. Run the conformance helper on every source type before
release. The frame, view, binding, planning, persistence, and hashing
machinery will then work without a single special case.
