# Describe a typed link between study representations

Describe a typed link between study representations

## Usage

``` r
frame_link(
  source,
  target,
  type = c("derivation", "mapping", "correspondence", "alignment"),
  map = NULL,
  source_axis = c("observation", "feature"),
  target_axis = c("observation", "feature"),
  metadata = list(),
  operator = NULL
)
```

## Arguments

- source:

  Source representation name.

- target:

  Target representation name.

- type:

  Link type: derivation, feature mapping, correspondence, or alignment.

- map:

  Optional scalar table with `.source_id` and `.target_id` columns.

- source_axis:

  Axis addressed by `.source_id`.

- target_axis:

  Axis addressed by `.target_id`.

- metadata:

  Serializable link metadata.

- operator:

  Optional typed feature operator. This is valid only for a
  feature-to-feature mapping or alignment and remains a first-class
  field.

## Value

A `frame_link` descriptor.

## Examples

``` r
link <- frame_link("bold", "surface", type = "derivation")
link$source
#> [1] "bold"
link$target
#> [1] "surface"
```
