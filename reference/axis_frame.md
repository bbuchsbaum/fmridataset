# Construct an annotated axis

Construct an annotated axis

## Usage

``` r
axis_frame(
  data,
  blocks = list(),
  id = NULL,
  axis = c("observation", "feature", "entity", "component"),
  id_col = NULL,
  metadata = list()
)

axis_data(x)

axis_blocks(x)

axis_ids(x)
```

## Arguments

- data:

  A data frame with one row per axis element.

- blocks:

  Named `axis_block` objects aligned on their first dimension.

- id:

  Optional stable IDs.

- axis:

  Axis role. Observation is the public default.

- id_col:

  Name of the ID column.

- metadata:

  Additional serializable metadata.

- x:

  An `axis_frame`.

## Value

An `axis_frame`.
