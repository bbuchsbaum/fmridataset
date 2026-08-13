# Resolve logical observation rows to shards

Resolve logical observation rows to shards

## Usage

``` r
locate_source_rows(x, observations = NULL)
```

## Arguments

- x:

  A `row_sharded_source`.

- observations:

  Logical observation positions in requested order.

## Value

A data frame mapping each request position to a shard and local row.
