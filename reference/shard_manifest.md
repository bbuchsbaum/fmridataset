# Inspect a row-sharded source manifest

Inspect a row-sharded source manifest

## Usage

``` r
shard_manifest(x)
```

## Arguments

- x:

  A `row_sharded_source`.

## Value

A data frame describing stable IDs, logical row ranges, source
fingerprints, and user-supplied shard metadata.

## Examples

``` r
shards <- list(memory_source(matrix(1:4, nrow = 2)), memory_source(matrix(5:8, nrow = 2)))
shard_manifest(row_sharded_source(shards))
#>      .shard_id .start .end .n_observation
#> 1 shard-000001      1    2              2
#> 2 shard-000002      3    4              2
#>                                                .source_fingerprint
#> 1 f1ed3d67cae82704ff0fa47dcbb70de41b5474b460c810bb1432e47d46250159
#> 2 d3b2749e1dbe8397c3db5a66d20f8599783e32a3861daf018295be840a778788
```
