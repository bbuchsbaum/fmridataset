# Reconstruct a study from semantic and physical components

Reconstruct a study from semantic and physical components

## Usage

``` r
study_from_fds_manifest(manifest, representations, bindings = list())
```

## Arguments

- manifest:

  A valid FDS v2 study manifest.

- representations:

  Named lazy frames or collections matching the representation
  manifests.

- bindings:

  Named physical bindings for shared study arrays.

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
manifest <- fds_study_manifest(study)
rebuilt <- study_from_fds_manifest(
  manifest, list(main = frame), fds_study_bindings(study)
)
study_ids(rebuilt)
#> [1] "main"
```
