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
