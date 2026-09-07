# Extract canonical study representations for persistence

Extract canonical study representations for persistence

## Usage

``` r
fds_study_representations(x)
```

## Arguments

- x:

  An `fmri_study`.

## Value

Named frames and collections matching `fds_study_manifest(x)`.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
study <- fmri_study(list(main = frame))
names(fds_study_representations(study))
#> [1] "main"
```
