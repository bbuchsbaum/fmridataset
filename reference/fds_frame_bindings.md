# Extract physical array bindings from a frame

Storage codecs use this helper to pair each source-free FDS array
declaration with its current physical or in-memory `array_source`.

## Usage

``` r
fds_frame_bindings(x)
```

## Arguments

- x:

  An `fmri_frame`.

## Value

A named list of `array_source` descriptors keyed exactly like the
manifest `arrays` registry.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
bindings <- fds_frame_bindings(frame)
names(bindings)
#> [1] "assays/bold"
```
