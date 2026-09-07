# Construct a linked fMRI study

Construct a linked fMRI study

## Usage

``` r
fmri_study(
  frames,
  entities = list(),
  links = list(),
  tables = list(),
  metadata = list(),
  provenance = NULL
)
```

## Arguments

- frames:

  Named `fmri_frame` or `fmri_collection` representations.

- entities:

  Shared authoritative entity registry.

- links:

  Named `frame_link` descriptors.

- tables:

  Named typed relational tables.

- metadata:

  Unaligned study-level metadata.

- provenance:

  `NULL` or a validated `provenance_graph`.

## Value

An `fmri_study`.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
study <- fmri_study(list(main = frame))
study_ids(study)
#> [1] "main"
```
