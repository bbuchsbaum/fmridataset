# Compute a stable relation-registry digest

Compute a stable relation-registry digest

## Usage

``` r
relation_registry_digest(x)
```

## Arguments

- x:

  A frame, view, or relation registry.

## Value

A hexadecimal digest over the normalized registry.

## Examples

``` r
registry <- relation_registry(
  observation_stimulus = key_relation("stimulus_id", target = "stimulus")
)
relation_registry_digest(registry)
#> [1] "bd8a931aaa915040f4444c0f85b840e39681d85db20fef091ab28b7999aa70dd"
```
