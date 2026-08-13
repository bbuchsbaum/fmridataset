# Derive the canonical map owned by a parent-linked target space

Parcel spaces contribute their aggregation operator and basis spaces
their analysis operator. Other transformations require an explicit
[`feature_map()`](https://bbuchsbaum.github.io/fmridataset/reference/feature_map.md).

## Usage

``` r
feature_map_from_target(target)
```

## Arguments

- target:

  A parent-linked `parcel_space` or `basis_space`.

## Value

A `feature_map` from `parent_space(target)` to `target`.
