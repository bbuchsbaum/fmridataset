# Access frame relations

Access frame relations

## Usage

``` r
relations(x, ...)

relation(x, name, ...)

relation_names(x)
```

## Arguments

- x:

  An `fmri_frame`, view, or relation registry.

- ...:

  Additional method arguments.

- name:

  One registered relation name.

## Value

`relations()` returns the registry; `relation()` returns one descriptor;
`relation_names()` returns registry names.

## Examples

``` r
registry <- relation_registry(
  observation_stimulus = key_relation("stimulus_id", target = "stimulus")
)
relations(registry)
#> <relation_registry> 1 relations
#>   observation_stimulus
relation(registry, "observation_stimulus")
#> $type
#> [1] "key"
#> 
#> $key
#> [1] "stimulus_id"
#> 
#> $source
#> [1] "observation"
#> 
#> $target
#> [1] "stimulus"
#> 
#> $allow_missing
#> [1] FALSE
#> 
#> $metadata
#> list()
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "key_relation"  "fmri_relation"
relation_names(registry)
#> [1] "observation_stimulus"
```
