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

  Named auxiliary tables.

- active_assay:

  Active assay name.

- metadata:

  Frame metadata.

- provenance:

  Serializable provenance records.

## Value

An `fmri_frame`.
