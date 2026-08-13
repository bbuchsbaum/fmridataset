# Validate and inspect feature maps

Validate and inspect feature maps

## Usage

``` r
validate_feature_map(x)

feature_map_source_space(x)

feature_map_target_space(x)

feature_map_operator(x)

feature_map_digest(x)
```

## Arguments

- x:

  A `feature_map`.

## Value

`validate_feature_map()` returns `x` invisibly. The accessors return the
source space, target space, linear operator, or deterministic digest.
