# Construct a lazy source transformed through a feature map

Construct a lazy source transformed through a feature map

## Usage

``` r
feature_mapped_source(source, map, rule = c("linear", "independent_variance"))
```

## Arguments

- source:

  Observation-by-source-feature `array_source`.

- map:

  A compatible `feature_map`.

- rule:

  Transformation rule. `"linear"` maps ordinary values;
  `"independent_variance"` maps diagonal variances with squared weights.

## Value

A serializable `feature_mapped_source`.
