# Describe an explicit sparse or many-to-many relation

Describe an explicit sparse or many-to-many relation

## Usage

``` r
sparse_relation(
  data,
  from,
  to,
  from_col = ".from_id",
  to_col = ".to_id",
  weight = NULL,
  directed = TRUE,
  metadata = list()
)
```

## Arguments

- data:

  Scalar edge table.

- from:

  Source domain.

- to:

  Target domain.

- from_col:

  Column containing source stable IDs.

- to_col:

  Column containing target stable IDs.

- weight:

  Optional numeric weight column.

- directed:

  Whether edge direction is semantically meaningful.

- metadata:

  Additional serializable metadata.

## Value

A serializable `sparse_relation` descriptor.

## Examples

``` r
edges <- sparse_relation(
  data = tibble::tibble(
    .from_id = c("obs-1", "obs-2"),
    .to_id = c("stim-1", "stim-2"),
    weight = c(0.7, 0.3)
  ),
  from = "observation",
  to = "entity:stimulus",
  weight = "weight"
)
edges$from
#> [1] "observation"
```
