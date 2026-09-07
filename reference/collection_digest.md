# Compute a deterministic collection digest

Compute a deterministic collection digest

## Usage

``` r
collection_digest(x)
```

## Arguments

- x:

  An `fmri_collection`.

## Value

A SHA-256 digest computed without reading numerical arrays.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
collection <- fmri_collection(list(sub01 = frame, sub02 = frame))
collection_digest(collection)
#> [1] "18d85dfb0e8f9cd5f42672c7977952a8ada1a739b3fb2c66f52ac1d60b4b985b"
```
