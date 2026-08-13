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
