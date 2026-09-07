# Derive the canonical map owned by a parent-linked target space

Parcel spaces contribute their aggregation operator and basis spaces
their analysis operator. Other transformations require an explicit
[`feature_map()`](https://bbuchsbaum.github.io/fmridataset/reference/feature_map.md).

## Usage

``` r
feature_map_from_target(target)
```

## Arguments

- target:

  A parent-linked `parcel_space` or `basis_space`.

## Value

A `feature_map` from `parent_space(target)` to `target`.

## Examples

``` r
parent <- volume_space(c(2, 2, 1), support = 1:4, template = "toy")
membership <- Matrix::sparseMatrix(
  i = 1:4, j = c(1L, 1L, 2L, 2L), x = 1, dims = c(4L, 2L)
)
target <- parcel_space(parent, c("left", "right"), membership, atlas = "toy")
m <- feature_map_from_target(target)
feature_map_target_space(m)
#> $parent
#> $dim
#> [1] 2 2 1
#> 
#> $affine
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
#> 
#> $support
#> [1] 1 2 3 4
#> 
#> $template
#> [1] "toy"
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
#> 
#> $parcel_ids
#> [1] "left"  "right"
#> 
#> $ids
#> [1] "toy:left"  "toy:right"
#> 
#> $membership
#> 4 x 2 sparse Matrix of class "dgCMatrix"
#>         
#> [1,] 1 .
#> [2,] 1 .
#> [3,] . 1
#> [4,] . 1
#> 
#> $aggregation_operator
#> 2 x 4 sparse Matrix of class "dgCMatrix"
#>                     
#> [1,] 0.5 0.5 .   .  
#> [2,] .   .   0.5 0.5
#> 
#> $decoder
#> 4 x 2 sparse Matrix of class "dgCMatrix"
#>         
#> [1,] 1 .
#> [2,] 1 .
#> [3,] . 1
#> [4,] . 1
#> 
#> $data
#> # A tibble: 2 × 2
#>   .feature_id id   
#>   <chr>       <chr>
#> 1 toy:left    left 
#> 2 toy:right   right
#> 
#> $atlas
#> $atlas$id
#> [1] "toy"
#> 
#> $atlas$n_parcels
#> [1] 2
#> 
#> 
#> $aggregation
#> [1] "mean"
#> 
#> $metadata
#> list()
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "parcel_space"  "feature_space"
```
