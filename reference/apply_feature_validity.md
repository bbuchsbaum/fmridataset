# Apply one validity relation lazily to frame assays

Apply one validity relation lazily to frame assays

## Usage

``` r
apply_feature_validity(x, name = NULL, assays = NULL)
```

## Arguments

- x:

  An `fmri_frame` or view.

- name:

  Validity relation name.

- assays:

  Assay names to mask. Defaults to all assays.

## Value

A new frame with lazy `NA` masking and derivation provenance.

## Examples

``` r
space <- index_space(4, ids = paste0("f", 1:4), namespace = "validity-ex")
subjects <- entity_frame(
  data.frame(subject_id = c("sub-1", "sub-2")),
  key = "subject_id"
)
validity <- entity_feature_validity(
  entity = "subject", entity_ids = c("sub-1", "sub-2"),
  masks = rbind(
    c(TRUE, TRUE, FALSE, TRUE),
    c(TRUE, FALSE, FALSE, TRUE)
  ),
  space = space
)
frame <- fmri_frame(
  assays = list(signal = matrix(1:8, nrow = 2)),
  observations = data.frame(.obs_id = c("o1", "o2"), subject_id = c("sub-1", "sub-2")),
  space = space,
  entities = list(subject = subjects),
  relations = list(
    observation_subject = key_relation("subject_id"),
    subject_feature_validity = validity
  )
)
masked_frame <- apply_feature_validity(frame)
collect_assay(masked_frame)
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    3   NA    7
#> [2,]    2   NA   NA    8
```
