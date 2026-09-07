# Upgrade a provisional FDS study manifest

Converts an FDS study v1 manifest, including its reverse-direction
provisional frame links, to the canonical v2 schema. Canonical v2
manifests are validated and returned unchanged.

## Usage

``` r
upgrade_fds_study_manifest(manifest)
```

## Arguments

- manifest:

  An FDS study v1 or v2 manifest.

## Value

A validated FDS study v2 manifest.

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
identical(upgrade_fds_study_manifest(manifest), manifest)
#> [1] TRUE
```
