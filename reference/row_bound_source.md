# Bind compatible sources along observations

Bind compatible sources along observations

## Usage

``` r
row_bound_source(sources)
```

## Arguments

- sources:

  A non-empty list of two-dimensional array sources.

## Value

A serializable `row_sharded_source`. This compatibility constructor
assigns deterministic shard IDs.

## Examples

``` r
a <- memory_source(matrix(1:4, nrow = 2))
b <- memory_source(matrix(5:8, nrow = 2))
src <- row_bound_source(list(a, b))
source_shape(src)
#> [1] 4 2
```
