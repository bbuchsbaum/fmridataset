# Construct a spatially typed annotated matrix

Construct a spatially typed annotated matrix

## Usage

``` r
fmri_frame(
  assays,
  observations,
  features = NULL,
  space = NULL,
  entities = list(),
  relations = list(),
  tables = list(),
  active_assay = NULL,
  metadata = list(),
  provenance = NULL
)
```

## Arguments

- assays:

  Named matrices or serializable array sources.

- observations:

  Observation metadata or an observation `axis_frame`.

- features:

  Feature metadata or a spatial feature axis.

- space:

  Feature space used when `features` is not already spatial.

- entities:

  A named `entity_registry` or entries normalizable by
  [`entity_registry()`](https://bbuchsbaum.github.io/fmridataset/reference/entity_registry.md).

- relations:

  Named relation registry.

- tables:

  Named typed tables created by
  [`event_table()`](https://bbuchsbaum.github.io/fmridataset/reference/event_table.md)
  or
  [`auxiliary_table()`](https://bbuchsbaum.github.io/fmridataset/reference/auxiliary_table.md).

- active_assay:

  Active assay name.

- metadata:

  Unaligned frame-level record. Aligned values belong on an axis,
  entity, block, assay, relation, typed table, or linked frame.

- provenance:

  `NULL` or a validated `provenance_graph`.

## Value

An `fmri_frame`.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
frame
#> <fmri_frame> 4 observations x 8 features
#>   assays: bold 
#>   active: bold 
#>   space: volume_space 9d51a33b4ccb 
```
