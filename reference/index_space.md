# Construct a generic indexed feature space

Construct a generic indexed feature space

## Usage

``` r
index_space(
  n,
  ids = NULL,
  namespace = NULL,
  data = NULL,
  id_policy = c("require", "deterministic", "ephemeral")
)
```

## Arguments

- n:

  Number of features.

- ids:

  Optional stable feature IDs.

- namespace:

  Stable namespace used for deterministic IDs.

- data:

  Optional feature metadata.

- id_policy:

  ID policy used when `ids` is absent. The default requires explicit
  IDs; deterministic IDs additionally require `namespace`.

## Value

An `index_space`.

## Examples

``` r
index_space(3, ids = c("a", "b", "c"))
#> $n
#> [1] 3
#> 
#> $ids
#> [1] "a" "b" "c"
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
#> # A tibble: 3 × 1
#>   .feature_id
#>   <chr>      
#> 1 a          
#> 2 b          
#> 3 c          
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "index_space"   "feature_space"
index_space(2, namespace = "roi", id_policy = "deterministic")
#> $n
#> [1] 2
#> 
#> $ids
#> [1] "feature-roi-000001" "feature-roi-000002"
#> 
#> $namespace
#> [1] "roi"
#> 
#> $id_policy
#> $policy
#> [1] "deterministic"
#> 
#> $namespace
#> [1] "roi"
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
#> 1 feature-roi-000001
#> 2 feature-roi-000002
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "index_space"   "feature_space"
```
