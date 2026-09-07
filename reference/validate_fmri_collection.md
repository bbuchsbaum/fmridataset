# Validate an fMRI collection

Validate an fMRI collection

## Usage

``` r
validate_fmri_collection(x)
```

## Arguments

- x:

  An `fmri_collection`.

## Value

`x`, invisibly, or a structured collection error.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
collection <- fmri_collection(list(sub01 = frame, sub02 = frame))
validate_fmri_collection(collection)
```
