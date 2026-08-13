# Construct and validate an entity registry

Construct and validate an entity registry

## Usage

``` r
entity_registry(entities = list(), ...)

validate_entity_registry(x)
```

## Arguments

- entities:

  A named list of `entity_frame` objects. For evolutionary
  compatibility, a named legacy entry with `data`, `blocks`, and either
  `key` or a conventional `<name>_id` column is normalized immediately.

- ...:

  Alternatively, named `entity_frame` objects.

- x:

  An entity registry.

## Value

A named `entity_registry`.
