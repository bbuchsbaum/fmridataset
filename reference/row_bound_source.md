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
