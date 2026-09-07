# Inspect parent-linked feature spaces and parcel-space operators

Inspect parent-linked feature spaces and parcel-space operators

## Usage

``` r
parent_space(x)

parcel_membership(x)

parcel_aggregation(x)
```

## Arguments

- x:

  A parent-linked feature space such as a `parcel_space` or
  `basis_space`.

## Value

`parent_space()` returns the parent feature space; `parcel_membership()`
and `parcel_aggregation()` return sparse operators.

## Examples

``` r
parent <- volume_space(c(3, 2, 1), support = 1:6)
membership <- Matrix::sparseMatrix(
  i = 1:6, j = c(1, 1, 1, 2, 2, 2), x = 1, dims = c(6L, 2L)
)
x <- parcel_space(parent, c(10L, 20L), membership, atlas = "toy-atlas")
parent_space(x)
#> $dim
#> [1] 3 2 1
#> 
#> $affine
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
#> 
#> $support
#> [1] 1 2 3 4 5 6
#> 
#> $template
#> NULL
#> 
#> $units
#> [1] "mm"
#> 
#> $metadata
#> list()
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "volume_space"  "feature_space"
parcel_membership(x)
#> 6 x 2 sparse Matrix of class "dgCMatrix"
#>         
#> [1,] 1 .
#> [2,] 1 .
#> [3,] 1 .
#> [4,] . 1
#> [5,] . 1
#> [6,] . 1
```
