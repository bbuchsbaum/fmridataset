# Compute a canonical FDS study-manifest digest

Compute a canonical FDS study-manifest digest

## Usage

``` r
fds_study_manifest_digest(manifest)
```

## Arguments

- manifest:

  A valid FDS study manifest.

## Value

A stable hexadecimal digest over source-free study semantics.

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
fds_study_manifest_digest(manifest)
#> [1] "6e572d1145cc8acdd210fa9c09b2c72912d26cc7c64c0b43e4e03ae1a42406d6"
```
