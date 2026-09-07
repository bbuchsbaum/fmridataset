# Access entities from a frame or registry

Access entities from a frame or registry

## Usage

``` r
entities(x, ...)

entity(x, name, ...)

entity_names(x)
```

## Arguments

- x:

  An `fmri_frame`, view, or `entity_registry`.

- ...:

  Additional method arguments.

- name:

  One registered entity name.

## Value

`entities()` returns the registry; `entity()` returns one
`entity_frame`; `entity_names()` returns registry names.

## Examples

``` r
subjects <- entity_frame(
  data = tibble::tibble(subject_id = c("sub-1", "sub-2")),
  key = "subject_id"
)
registry <- entity_registry(subject = subjects)
entities(registry)
#> <entity_registry> 1 entity types
#>   subject
entity(registry, "subject")
#> <entity_frame> 2 entity records
#>   key: subject_id 
entity_names(registry)
#> [1] "subject"
```
