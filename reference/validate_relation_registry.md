# Validate a relation registry

Validate a relation registry

## Usage

``` r
validate_relation_registry(
  x,
  observations = NULL,
  features = NULL,
  entities = NULL
)
```

## Arguments

- x:

  A `relation_registry`.

- observations:

  Optional observation `axis_frame`.

- features:

  Optional feature `axis_frame`.

- entities:

  Optional `entity_registry`.

## Value

Invisibly returns `x`; contextual validation also enforces all
foreign-key and edge identities.
