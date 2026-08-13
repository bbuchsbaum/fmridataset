# Describe a typed link between study representations

Describe a typed link between study representations

## Usage

``` r
frame_link(
  from,
  to,
  type = c("derived_from", "mapped_from", "corresponds_to", "aligned_from"),
  map = NULL,
  from_axis = c("observation", "feature"),
  to_axis = c("observation", "feature"),
  metadata = list(),
  feature_map = NULL
)
```

## Arguments

- from:

  Source representation name.

- to:

  Target representation name.

- type:

  Link type: derivation, feature mapping, correspondence, or alignment.

- map:

  Optional scalar table with `.from_id` and `.to_id` columns.

- from_axis:

  Axis addressed by `.from_id`.

- to_axis:

  Axis addressed by `.to_id`.

- metadata:

  Serializable link metadata.

- feature_map:

  Optional typed feature map. This is valid only for a
  feature-to-feature `"mapped_from"` link and is persisted in the link's
  metadata without changing the v1 descriptor shape.

## Value

A `frame_link` descriptor.
