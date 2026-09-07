# Inspect typed identity domains

Semantic and schema identities are independent of physical locations.
Source fingerprints identify descriptors and selectors, not array
contents. Content digests are optional receipts computed explicitly by
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
or supplied by a backend; this function never reads data to infer one,
so the content domain always requires `content_digest`.

## Usage

``` r
identity_descriptor(
  x,
  domain = c("auto", "semantic", "schema", "space", "source", "provenance", "content"),
  content_digest = NULL
)
```

## Arguments

- x:

  A frame, frame schema, feature space, array source, FDS manifest,
  provenance graph, collection, or study.

- domain:

  Identity domain. Usually inferred from `x`.

- content_digest:

  Optional externally computed content digest, such as the value of
  [`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md).

## Value

A serializable typed identity descriptor.

## Examples

``` r
src <- memory_source(matrix(seq_len(6), nrow = 2))
identity_descriptor(src)$domain
#> [1] "source"
```
