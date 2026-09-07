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

## Examples

``` r
parts <- list(
  left = index_space(2, ids = c("l1", "l2")),
  right = index_space(2, ids = c("r1", "r2"))
)
x <- composite_space(parts, composite_type = "bilateral")
feature_ids(x)
#> [1] "left::l1"  "left::l2"  "right::r1" "right::r2"
```
