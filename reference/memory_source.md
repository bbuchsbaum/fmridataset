# Construct an in-memory array source

Construct an in-memory array source

## Usage

``` r
memory_source(data, dtype = NULL, chunks = NULL)
```

## Arguments

- data:

  A two-dimensional matrix or array.

- dtype:

  Logical storage dtype. Numeric R matrices default to `"float64"`.

- chunks:

  Optional logical chunk shape.

## Value

A serializable `memory_source`.
