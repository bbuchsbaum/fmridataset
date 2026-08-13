# Construct an ordered composite feature space

A `composite_space` forms one feature axis from heterogeneous child
spaces, such as left cortex, right cortex, and subcortical volume. It
owns only the ordered routing between that axis and its named parts;
each child remains the authority for spatial identity, vectorization,
and reconstruction.

## Usage

``` r
composite_space(
  parts,
  composite_type = "composite",
  metadata = list(),
  route = NULL
)
```

## Arguments

- parts:

  A named list of non-empty `feature_space` objects.

- composite_type:

  A stable semantic label, such as `"grayordinate_like"`.

- metadata:

  Additional serializable metadata.

- route:

  Optional internal routing table with `part` and `part_index` columns.
  By default, all child features are concatenated in part order.

## Value

A `composite_space`.
