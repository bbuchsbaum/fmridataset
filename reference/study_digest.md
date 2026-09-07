# Compute a deterministic study digest

Compute a deterministic study digest

## Usage

``` r
study_digest(x)
```

## Arguments

- x:

  An `fmri_study` or filtered view.

## Value

A SHA-256 digest computed without numerical reads.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
study <- fmri_study(list(main = frame))
study_digest(study)
#> [1] "d8435b6df0f7eb8f4e0f7c35deabc41bc22d87627d3a9171dc0f04727588e98c"
```
