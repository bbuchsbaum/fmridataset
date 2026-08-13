# Append immutable source shards

Append immutable source shards

## Usage

``` r
append_source_shards(x, sources, shard_ids = NULL, shard_data = NULL)
```

## Arguments

- x:

  An existing `row_sharded_source`.

- sources:

  New compatible child sources.

- shard_ids:

  Stable IDs for the new shards.

- shard_data:

  Optional metadata for the new shards. Its columns must match existing
  shard metadata.

## Value

A new `row_sharded_source`; `x` and its child descriptors are not
modified.
