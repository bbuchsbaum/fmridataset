# Convert an array source to a lazy delarr array

`as_delarr()` wraps a serializable [array
source](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
as a `delarr` provider so that bounded, chunk-aware execution can be
delegated to `delarr` without materializing the assay. The realization
budget is enforced before any provider is created.

## Usage

``` r
as_delarr(x, memory_budget = Inf, ...)

# Default S3 method
as_delarr(x, memory_budget = Inf, ...)
```

## Arguments

- x:

  An array source, or another object with an `as_delarr()` method.

- memory_budget:

  Maximum realized bytes permitted for a single pull. `Inf` disables the
  check.

- ...:

  Additional arguments passed to methods.

## Value

A `delarr` lazy array whose pulls route through
[`source_read()`](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md).

## See also

[array-source](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
for the source protocol.

## Examples

``` r
src <- memory_source(matrix(seq_len(6), nrow = 2))
lazy <- as_delarr(src)
dim(lazy)
#> [1] 2 3
```
