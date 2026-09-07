# Validate and inspect feature maps

Validate and inspect feature maps

## Usage

``` r
validate_feature_map(x)

feature_map_source_space(x)

feature_map_target_space(x)

feature_map_operator(x)

feature_map_digest(x)
```

## Arguments

- x:

  A `feature_map`.

## Value

`validate_feature_map()` returns `x` invisibly. The accessors return the
source space, target space, linear operator, or deterministic digest.

## Examples

``` r
src <- index_space(4, ids = paste0("v", 1:4), namespace = "map-source")
tgt <- index_space(2, ids = paste0("p", 1:2), namespace = "map-target")
op <- matrix(c(0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5), nrow = 2, byrow = TRUE)
m <- feature_map(src, tgt, op, map_type = "toy_aggregation")
feature_map_source_space(m)
#> $n
#> [1] 4
#> 
#> $ids
#> [1] "v1" "v2" "v3" "v4"
#> 
#> $namespace
#> [1] "map-source"
#> 
#> $id_policy
#> $policy
#> [1] "require"
#> 
#> $namespace
#> [1] "map-source"
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
#> # A tibble: 4 × 1
#>   .feature_id
#>   <chr>      
#> 1 v1         
#> 2 v2         
#> 3 v3         
#> 4 v4         
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "index_space"   "feature_space"
feature_map_target_space(m)
#> $n
#> [1] 2
#> 
#> $ids
#> [1] "p1" "p2"
#> 
#> $namespace
#> [1] "map-target"
#> 
#> $id_policy
#> $policy
#> [1] "require"
#> 
#> $namespace
#> [1] "map-target"
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
#> 1 p1         
#> 2 p2         
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "index_space"   "feature_space"
feature_map_digest(m)
#> [1] "1abd1d7b9be78a6171bccdb0f22b4de12ac3bec2fd90379ed2042913e9862fc8"
```
