# Compute the deterministic digest of a hierarchy index

Compute the deterministic digest of a hierarchy index

## Usage

``` r
hierarchy_digest(x)
```

## Arguments

- x:

  An `fmri_hierarchy_index`.

## Value

A SHA-256 digest.

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
hierarchy_digest(idx)
#> [1] "a5f813d815d6df1be3d12552e9b0129a3c6f30c078551cb5ad786f5a05a2d21d"
```
