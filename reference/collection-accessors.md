# Access frames in an fMRI collection

Access frames in an fMRI collection

## Usage

``` r
collection_frames(x)

collection_frame(x, id)

collection_ids(x)
```

## Arguments

- x:

  An `fmri_collection`.

- id:

  One stable frame ID.

## Value

The named frame list, one frame, or the stable frame IDs.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
collection <- fmri_collection(list(sub01 = frame, sub02 = frame))
collection_ids(collection)
#> [1] "sub01" "sub02"
collection_frame(collection, "sub01")
#> <fmri_frame> 3 observations x 4 features
#>   assays: bold 
#>   active: bold 
#>   space: volume_space ca8461c12469 
```
