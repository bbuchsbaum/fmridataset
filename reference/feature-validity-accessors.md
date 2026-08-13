# Validate and inspect entity-feature validity

Validate and inspect entity-feature validity

## Usage

``` r
validate_entity_feature_validity(x)

validity_entity(x, name = NULL)

validity_entity_ids(x, name = NULL)

validity_mask_bank(x, name = NULL)

validity_space(x, name = NULL)

validity_matrix(x, name = NULL)
```

## Arguments

- x:

  An `entity_feature_validity`, frame, or view.

- name:

  Relation name when `x` is a frame or view.

## Value

The validated descriptor, entity name/IDs, mask bank, feature space, or
expanded entity-by-feature logical matrix.
