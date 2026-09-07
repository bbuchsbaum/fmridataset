# Construct an in-memory array source

A memory source holds its values in the descriptor, so its revision
fingerprint cannot be derived from a file state. It is instead derived
from the shape, dtype, chunk grid, and a per-object identity token
assigned once at construction, plus the optional caller-supplied
`revision`. Construction therefore never reads or hashes the payload,
whatever its size.

## Usage

``` r
memory_source(
  data,
  dtype = NULL,
  chunks = NULL,
  revision = NULL,
  identity = c("object", "content")
)
```

## Arguments

- data:

  A two-dimensional matrix or array.

- dtype:

  Logical storage dtype. Numeric R matrices default to `"float64"`.

- chunks:

  Optional logical chunk shape.

- revision:

  Optional single string naming the revision of `data`, such as a
  version label or an upstream checksum. It is never interpreted, but it
  replaces the per-object identity token, so two sources built
  independently under the same revision share a fingerprint: the caller
  is asserting they are the same source in the same revision.

- identity:

  `"object"` assigns a fresh identity token (unless `revision` is
  supplied); `"content"` derives it from
  [`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
  of `data`.

## Value

A serializable `memory_source`.

## Details

Two independently constructed memory sources with equal values have
different fingerprints by design, and fingerprint equality is never
value equality. A serialization round trip preserves the token, so a
descriptor and its copies agree. Values compared across objects require
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md).
Set `identity = "content"` to opt into a content-derived token: the
payload is hashed once at construction and equal-valued sources then
share a fingerprint at O(n) cost.

## See also

[`source_fingerprint()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
and
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
for the fingerprint policy.

## Examples

``` r
src <- memory_source(matrix(seq_len(6), nrow = 2))
source_shape(src)
#> [1] 2 3
source_dtype(src)
#> [1] "float64"
```
