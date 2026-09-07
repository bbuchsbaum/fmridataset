# Reconstruct a frame from an FDS manifest and physical sources

Reconstruct a frame from an FDS manifest and physical sources

## Usage

``` r
frame_from_fds_manifest(manifest, bindings)
```

## Arguments

- manifest:

  A valid FDS v1 frame manifest.

- bindings:

  Named physical array payloads or `array_source` descriptors, one per
  manifest array declaration.

## Value

An `fmri_frame` whose semantic state comes from `manifest` and whose
lazy arrays come from `bindings`.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
manifest <- fds_frame_manifest(frame)
bindings <- fds_frame_bindings(frame)
rebuilt <- frame_from_fds_manifest(manifest, bindings)
identical(feature_ids(rebuilt), feature_ids(frame))
#> [1] TRUE
```
