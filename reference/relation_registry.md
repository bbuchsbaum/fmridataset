# Construct a relation registry

Construct a relation registry

## Usage

``` r
relation_registry(relations = list(), ...)
```

## Arguments

- relations:

  A named list of `key_relation` or `sparse_relation` descriptors.

- ...:

  Alternatively, named relation descriptors.

## Value

A named `relation_registry`.

## Examples

``` r
registry <- relation_registry(
  observation_stimulus = key_relation("stimulus_id", target = "stimulus")
)
relation_names(registry)
#> [1] "observation_stimulus"
```
