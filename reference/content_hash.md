# Compute a numerical content hash of an array source

`content_hash()` is the one operation in the package that identifies an
array by its values. It is the content domain of the identity contract
(see
[`identity_descriptor()`](https://bbuchsbaum.github.io/fmridataset/reference/identity_descriptor.md))
and it is always explicit: nothing in the package computes it for you.
Ordinary construction, inspection, fingerprinting, planning, and
persistence never call it.

## Usage

``` r
content_hash(x, ...)

# Default S3 method
content_hash(x, ...)

# S3 method for class 'array_source'
content_hash(
  x,
  ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
)

# S3 method for class 'memory_source'
content_hash(
  x,
  ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
)

# S3 method for class 'source_view'
content_hash(
  x,
  ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
)

# S3 method for class 'row_bound_source'
content_hash(
  x,
  ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
)
```

## Arguments

- x:

  An array source, a handle, or a matrix coercible to a memory source.

- ...:

  Passed to methods.

- block_bytes:

  Upper bound on the realized bytes of one read. The default reuses the
  block-planning option `fmridataset.target_block_bytes`.

## Value

One lowercase SHA-256 hexadecimal string.

## Details

The digest is `O(n)` in the number of values. The default method
streams: it opens one handle, reads bounded row blocks through the
source protocol aligned to the source chunk grid, and hashes each block
into a running state, so no source is materialized whole. `block_bytes`
bounds the realized size of one read and affects only I/O; every block
size produces the same digest. Blocks cover whole rows when a row fits
the bound and consecutive feature ranges of one row otherwise.

Two sources hash equal exactly when they have the same shape, realize to
the same R mode (double, logical, or complex), and hold equal values.
Storage dtype, chunking, wrappers, views, and shard composition do not
enter the digest, so an in-memory copy and a row-bound composition of
the same values agree. `NaN` payloads and `NA` are normalized to one
representation each and remain distinct; negative zero stays distinct
from zero.

The digest describes the values at the moment they were read. It is a
receipt, not a live property: a later in-memory mutation of a memory
source's payload is not detected by
[`source_fingerprint()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
and shows only when a fresh `content_hash()` is compared with the
earlier one.

## See also

[`source_fingerprint()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
for the cheap revision identity of a descriptor,
[`identity_descriptor()`](https://bbuchsbaum.github.io/fmridataset/reference/identity_descriptor.md)
for typed identity domains, and
`inst/architecture/ADR-009-source-fingerprints-and-content-hashes.md`
for the policy.

## Examples

``` r
a <- memory_source(matrix(1:6, 2, 3))
b <- memory_source(matrix(as.double(1:6), 2, 3), chunks = c(1, 1))
identical(source_fingerprint(a), source_fingerprint(b))
#> [1] FALSE
identical(content_hash(a), content_hash(b))
#> [1] TRUE
```
