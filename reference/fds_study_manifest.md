# Construct and validate an FDS v2 study manifest

Study manifests retain shared entities, typed links, relational tables,
and the semantic manifests of every frame or collection member.
Numerical sources remain separate bindings so physical storage packages
do not own or reinterpret study semantics.

## Usage

``` r
fds_study_manifest(x)

validate_fds_study_manifest(manifest)
```

## Arguments

- x:

  An `fmri_study`.

- manifest:

  An FDS study manifest.

## Value

`fds_study_manifest()` returns a serializable source-free manifest;
`validate_fds_study_manifest()` returns `manifest` invisibly.

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
names(manifest$representations)
#> [1] "main"
```
