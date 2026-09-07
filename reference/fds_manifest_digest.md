# Compute a canonical FDS manifest digest

Compute a canonical FDS manifest digest

## Usage

``` r
fds_manifest_digest(manifest)
```

## Arguments

- manifest:

  A valid FDS manifest.

## Value

A stable hexadecimal digest over semantic manifest content.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
manifest <- fds_frame_manifest(frame)
fds_manifest_digest(manifest)
#> [1] "8902185b6cb89f4bdc9301c4d24f621ef65a8e3550cb1322ea6a9dcb65144e88"
```
