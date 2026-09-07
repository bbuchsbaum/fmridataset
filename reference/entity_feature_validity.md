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

## Examples

``` r
space <- index_space(4, ids = paste0("f", 1:4), namespace = "validity-ex")
validity <- entity_feature_validity(
  entity = "subject", entity_ids = c("sub-1", "sub-2"),
  masks = rbind(
    c(TRUE, TRUE, FALSE, TRUE),
    c(TRUE, FALSE, FALSE, TRUE)
  ),
  space = space
)
validity_entity(validity)
#> [1] "subject"
```
