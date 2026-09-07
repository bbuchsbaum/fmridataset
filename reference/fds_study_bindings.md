# Extract shared study-level physical bindings

Extract shared study-level physical bindings

## Usage

``` r
fds_study_bindings(x)
```

## Arguments

- x:

  An `fmri_study`.

## Value

A named list of shared entity-block payloads. Representation arrays
remain owned by their individual frame bindings.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
study <- fmri_study(list(main = frame))
fds_study_bindings(study)
#> list()
```
