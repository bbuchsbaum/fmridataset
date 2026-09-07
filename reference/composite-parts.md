# Inspect composite feature-space parts

Inspect composite feature-space parts

## Usage

``` r
composite_parts(x)

composite_part_names(x)

composite_part(x, name)
```

## Arguments

- x:

  A `composite_space`.

- name:

  One child part name.

## Value

`composite_parts()` returns the ordered named child spaces;
`composite_part_names()` returns their names; and `composite_part()`
returns one child space.

## Examples

``` r
parts <- list(
  left = index_space(2, ids = c("l1", "l2")),
  right = index_space(2, ids = c("r1", "r2"))
)
x <- composite_space(parts)
composite_part_names(x)
#> [1] "left"  "right"
composite_part(x, "left")
#> $n
#> [1] 2
#> 
#> $ids
#> [1] "l1" "l2"
#> 
#> $namespace
#> NULL
#> 
#> $id_policy
#> $policy
#> [1] "require"
#> 
#> $namespace
#> NULL
#> 
#> $keys
#> character(0)
#> 
#> $durable
#> [1] TRUE
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "fmri_id_policy" "list"          
#> 
#> $data
#> # A tibble: 2 × 1
#>   .feature_id
#>   <chr>      
#> 1 l1         
#> 2 l2         
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "index_space"   "feature_space"
```
