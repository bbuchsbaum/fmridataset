# Inspect axis ID durability

Inspect axis ID durability

## Usage

``` r
axis_id_policy(x)

ids_are_durable(x)
```

## Arguments

- x:

  An axis, feature space, frame, or view.

## Value

`axis_id_policy()` returns the versioned ID-policy descriptor;
`ids_are_durable()` returns one logical value.

## Examples

``` r
x <- axis_frame(data.frame(value = 1:3), id_policy = "ephemeral")
axis_id_policy(x)
#> $policy
#> [1] "ephemeral"
#> 
#> $namespace
#> NULL
#> 
#> $keys
#> character(0)
#> 
#> $durable
#> [1] FALSE
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "fmri_id_policy" "list"          
ids_are_durable(x)
#> [1] FALSE
```
