# Construct an axis-aligned multivariate block

Construct an axis-aligned multivariate block

## Usage

``` r
axis_block(
  data,
  components = NULL,
  role = "continuous",
  units = NULL,
  metadata = list()
)

axis_block_data(x)

block_components(x)

block_component_ids(x)
```

## Arguments

- data:

  A matrix, array, lazy array, or serializable array source. Its first
  dimension is aligned with the owning axis.

- components:

  Component metadata. The `.component_id` column is generated when
  absent.

- role:

  Semantic role such as `"continuous"`, `"confound"`, or `"embedding"`.

- units:

  Optional units applying to the block as a whole.

- metadata:

  Additional serializable metadata.

- x:

  An `axis_block`.

## Value

An `axis_block`.
