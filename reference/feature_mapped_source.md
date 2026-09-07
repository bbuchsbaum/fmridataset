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

## Examples

``` r
src <- index_space(4, ids = paste0("v", 1:4), namespace = "map-source")
tgt <- index_space(2, ids = paste0("p", 1:2), namespace = "map-target")
op <- matrix(c(0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5), nrow = 2, byrow = TRUE)
m <- feature_map(src, tgt, op, map_type = "toy_aggregation")
fs <- feature_mapped_source(memory_source(matrix(1:12, nrow = 3)), m)
dim(source_read(fs))
#> [1] 3 2
```
