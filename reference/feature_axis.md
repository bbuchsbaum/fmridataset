# Construct a spatial feature axis

Construct a spatial feature axis

## Usage

``` r
feature_axis(data, space = NULL, blocks = list(), metadata = list(), ...)
```

## Arguments

- data:

  Feature metadata or an `fmri_frame` when used as an accessor.

- space:

  A `FeatureSpace`.

- blocks:

  Feature-aligned blocks.

- metadata:

  Additional metadata.

- ...:

  Additional arguments for methods.

## Value

A feature `axis_frame` carrying its space.

## Examples

``` r
sp <- index_space(3, ids = sprintf("f%d", 1:3), namespace = "ex")
fx <- feature_axis(data.frame(.feature_id = sprintf("f%d", 1:3)), space = sp)
axis_ids(fx)
#> [1] "f1" "f2" "f3"
```
