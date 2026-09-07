# Derive stable observation hierarchy indices

A hierarchy index is an immutable, assay-free cache derived from
validated key relations. `levels` are supplied root-to-leaf. Each
adjacent edge must form a strict chain from observations to the deepest
entity and then through its parents. Crossed relations are therefore
never mistaken for containment.

## Usage

``` r
hierarchy_index(x, levels, relations = NULL)
```

## Arguments

- x:

  An `fmri_frame` or `fmri_view`.

- levels:

  Unique entity names in root-to-leaf order.

- relations:

  Optional named character vector mapping every level to the
  key-relation name used for its incoming edge. This is required when an
  edge is ambiguous.

## Value

An `fmri_hierarchy_index`.

## Details

Integer group codes use entity-registry order, so they remain stable
when a frame is filtered or reordered.

## Examples

``` r
subject <- entity_frame(
  data = tibble::tibble(subject_id = c("sub-1", "sub-2")),
  key = "subject_id"
)
observations <- tibble::tibble(
  .obs_id = paste0("obs-", 1:4),
  subject_id = c("sub-1", "sub-1", "sub-2", "sub-2")
)
frame <- fmri_frame(
  assays = list(beta = memory_source(matrix(as.double(1:12), 4, 3))),
  observations = observations,
  entities = list(subject = subject),
  relations = list(
    observation_subject = key_relation("subject_id", target = "subject")
  )
)
idx <- hierarchy_index(frame, levels = "subject")
hierarchy_ids(idx)
#> # A tibble: 4 × 2
#>   .obs_id subject
#>   <chr>   <chr>  
#> 1 obs-1   sub-1  
#> 2 obs-2   sub-1  
#> 3 obs-3   sub-2  
#> 4 obs-4   sub-2  
```
