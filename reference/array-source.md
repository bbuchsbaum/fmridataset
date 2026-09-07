# Serializable numerical array sources

The estimate distinguishes source storage from the R object returned by
a read. Numeric source dtypes, including float16 and float32, are
realized as R doubles. The conservative peak estimate adds selection,
conversion, and compressed-input buffers to the retained output. It
covers numerical payloads; fixed R object headers and selector metadata
are outside the estimate.

## Usage

``` r
as_array_source(x, ...)

source_shape(x, ...)

source_dtype(x, ...)

source_chunks(x, ...)

source_capabilities(x, ...)

source_fingerprint(x, ...)

source_open(x, ...)

source_read(x, observations = NULL, features = NULL, ...)

source_read_native(x, observations = NULL, ...)

source_close(x, ...)

source_realization_cost(x, observations = NULL, features = NULL)
```

## Arguments

- x:

  An array source or object coercible to one.

- ...:

  Additional method arguments.

- observations:

  Optional observation selector: `NULL` for every observation, integer
  positions in request order, or a logical mask. Selectors follow the
  package's normalization law: positions must be whole numbers, may
  reorder, may be negative but not mixed with positive, drop zero, must
  be in bounds, and may not repeat an element; masks must match the axis
  length with no `NA`; an empty selection is legal. The law is checked
  at the generic before a method is dispatched.

- features:

  Optional feature selector, under the same law.

## Value

A `source_realization_cost` list containing storage and realized dtypes,
storage and output bytes, temporary buffer components, and the estimated
peak bytes.

## Details

**Fingerprint policy.** `source_fingerprint()` returns the revision
fingerprint of a descriptor: one SHA-256 string over the serializable
descriptor and the physical revision evidence a backend can observe
without reading values (file sizes and modification times, store
metadata, or a per-object identity token for in-memory sources),
combined with selectors and shard composition for wrappers. It never
hashes array values, it is computed once at construction and cached in
the descriptor, and it survives serialization. Equal fingerprints mean
"the same descriptor of the same revision"; they never mean equal
values, and different fingerprints never mean different values. Two
independently constructed
[`memory_source()`](https://bbuchsbaum.github.io/fmridataset/reference/memory_source.md)
objects with equal payloads have different fingerprints by design.

Value identity is a separate, explicit, `O(n)` operation:
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md).
File-backed sources compare their captured revision evidence with the
physical state before every open and read and raise
`fmridataset_error_source_stale` when it differs; genuine backend
failures raise `fmridataset_error_backend_io`. The policy is recorded in
`inst/architecture/ADR-009-source-fingerprints-and-content-hashes.md`.

## Examples

``` r
src <- memory_source(matrix(seq_len(6), nrow = 2))
source_shape(src)
#> [1] 2 3
source_dtype(src)
#> [1] "float64"
handle <- source_open(src)
source_read(handle, observations = 1)
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
source_close(handle)
```
