# Lazily transform a frame into a new feature domain

Lazily transform a frame into a new feature domain

## Usage

``` r
map_features(x, target = NULL, map = NULL, assay_rules = "linear")
```

## Arguments

- x:

  An `fmri_frame` or view.

- target:

  Optional parent-linked target space from which a canonical map can be
  derived.

- map:

  Optional explicit `feature_map`.

- assay_rules:

  Named rules for every assay: `"linear"` or `"independent_variance"`.
  Unnamed scalar rules are recycled.

## Value

A new linked-domain `fmri_frame` whose assays remain lazy.
