# Describe an explicit transformation between feature spaces

A feature map owns a target-by-source linear operator and the complete
spatial identity of both axes. Equal dimensions are never treated as
evidence of spatial compatibility. Statistical execution plans and
covariance models remain the responsibility of packages such as
`fmrigds`.

## Usage

``` r
feature_map(
  from,
  to,
  operator,
  map_type = "linear",
  traits = list(linear = TRUE),
  provenance = list(),
  metadata = list()
)
```

## Arguments

- from:

  Source `feature_space`.

- to:

  Target `feature_space`.

- operator:

  Target-by-source matrix, sparse `Matrix`, or serializable
  two-dimensional `array_source`.

- map_type:

  Stable map-family label.

- traits:

  Named serializable semantic traits.

- provenance:

  Serializable derivation metadata for the map itself.

- metadata:

  Additional serializable metadata.

## Value

A serializable `feature_map` descriptor.

## Examples

``` r
src <- index_space(4, ids = paste0("v", 1:4), namespace = "map-source")
tgt <- index_space(2, ids = paste0("p", 1:2), namespace = "map-target")
op <- matrix(c(0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5), nrow = 2, byrow = TRUE)
m <- feature_map(src, tgt, op, map_type = "toy_aggregation")
feature_map_operator(m)
#>      [,1] [,2] [,3] [,4]
#> [1,]  0.5  0.5  0.0  0.0
#> [2,]  0.0  0.0  0.5  0.5
```
