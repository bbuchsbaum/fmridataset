# Attach rowData metadata to a lazy matrix

Helper for reattaching metadata after DelayedMatrixStats operations.

## Usage

``` r
with_rowData(x, rowData)
```

## Arguments

- x:

  A lazy matrix or matrix-like object

- rowData:

  A data.frame of row-wise metadata

## Value

`x` with `rowData` attribute set
