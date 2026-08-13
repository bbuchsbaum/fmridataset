# Describe compressed entity-by-feature validity

Describe compressed entity-by-feature validity

## Usage

``` r
entity_feature_validity(entity, entity_ids, masks, space, metadata = list())
```

## Arguments

- entity:

  Entity registry name.

- entity_ids:

  Stable entity IDs aligned to mask rows.

- masks:

  Logical entity-by-feature matrix or a `mask_bank` whose original row
  assignments have the same length as `entity_ids`.

- space:

  Exact feature space addressed by validity columns.

- metadata:

  Serializable relation metadata.

## Value

An `entity_feature_validity` relation descriptor.
