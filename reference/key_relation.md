# Describe a symbolic foreign-key relation

A key relation does not duplicate a mapping. It declares that one scalar
column on an observation, feature, or entity table references the stable
IDs of one entity frame.

## Usage

``` r
key_relation(
  key,
  target = NULL,
  source = "observation",
  allow_missing = FALSE,
  metadata = list()
)
```

## Arguments

- key:

  Foreign-key column on the source domain.

- target:

  Target entity name. When `NULL`, frame validation infers the unique
  entity whose primary key has the same name as `key`.

- source:

  Source domain: `"observation"`, `"feature"`, an entity name, or an
  explicit `"entity:<name>"` domain.

- allow_missing:

  Whether missing foreign-key values are permitted.

- metadata:

  Additional serializable metadata.

## Value

A serializable `key_relation` descriptor.
