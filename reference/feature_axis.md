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
