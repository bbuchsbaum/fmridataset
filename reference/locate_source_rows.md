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

## Examples

``` r
shards <- list(memory_source(matrix(1:4, nrow = 2)), memory_source(matrix(5:8, nrow = 2)))
locate_source_rows(row_sharded_source(shards), observations = c(1, 3))
#>   .request_position .observation .shard_index    .shard_id .local_observation
#> 1                 1            1            1 shard-000001                  1
#> 2                 2            3            2 shard-000002                  1
```
