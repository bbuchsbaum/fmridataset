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
