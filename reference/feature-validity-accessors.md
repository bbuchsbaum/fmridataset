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
validity_entity_ids(validity)
#> [1] "sub-1" "sub-2"
validity_matrix(validity)
#>      [,1]  [,2]  [,3] [,4]
#> [1,] TRUE  TRUE FALSE TRUE
#> [2,] TRUE FALSE FALSE TRUE
```
