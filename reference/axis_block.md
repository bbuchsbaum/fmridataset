# Construct an axis-aligned multivariate block

A block is two-dimensional: rows are the elements of the owning axis and
columns are named components. Arrays with more than two dimensions are
rejected because a trailing axis without typed metadata would be an
anonymous semantic dimension; represent higher-order structure as named
components, as several blocks, or as an assay.

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

  A matrix, two-dimensional lazy array, or serializable array source.
  Its first dimension is aligned with the owning axis and its second
  dimension indexes `components`.

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

## Examples

``` r
b <- axis_block(
  matrix(1:6, 3, 2),
  components = data.frame(.component_id = c("x", "y")),
  role = "continuous"
)
dim(axis_block_data(b))
#> [1] 3 2
block_component_ids(b)
#> [1] "x" "y"
```
