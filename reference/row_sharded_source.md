# Construct a manifest-backed row-sharded source

`row_sharded_source()` presents compatible child sources as one logical
observation-by-feature array. Stable shard IDs and explicit boundaries
make global-to-local row routing inspectable and serializable. Reads are
grouped by touched shard, so an arbitrary ordered selector is issued at
most once to each selected child.

## Usage

``` r
row_sharded_source(sources, shard_ids = NULL, shard_data = NULL)
```

## Arguments

- sources:

  A non-empty list of compatible two-dimensional array sources.

- shard_ids:

  Stable, unique shard identifiers.

- shard_data:

  Optional scalar metadata with one row per shard. Names used by the
  shard manifest are reserved.

## Value

A serializable `row_sharded_source`.
