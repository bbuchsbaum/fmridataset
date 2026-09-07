# Filter every study representation through one shared entity selection

Filter every study representation through one shared entity selection

## Usage

``` r
filter_entities(x, entity, predicate)
```

## Arguments

- x:

  An `fmri_study`.

- entity:

  Bare or quoted shared entity name.

- predicate:

  A scalar-metadata predicate evaluated on that entity table.

## Value

A self-contained `fmri_study` whose numerical sources remain lazy.

## Examples

``` r
subject_entity <- entity_frame(
  data.frame(subject_id = c("s1", "s2"), age = c(20, 60)),
  key = "subject_id"
)
obs <- data.frame(
  .obs_id = paste0("o", 1:4),
  subject_id = c("s1", "s1", "s2", "s2")
)
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(16), nrow = 4)),
  observations = obs,
  space = index_space(4, ids = paste0("f", 1:4)),
  entities = list(subject = subject_entity),
  relations = list(
    observation_subject = key_relation("subject_id", target = "subject")
  )
)
study <- fmri_study(list(main = frame), entities = list(subject = subject_entity))
older <- filter_entities(study, subject, age > 30)
observation_ids(study_frame(older, "main"))
#> [1] "o3" "o4"
```
