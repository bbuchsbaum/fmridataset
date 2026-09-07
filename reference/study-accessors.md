# Study representation accessors

Study representation accessors

## Usage

``` r
study_frames(x, contextual = TRUE)

study_frame(x, name, contextual = TRUE)

study_ids(x)
```

## Arguments

- x:

  An `fmri_study`.

- contextual:

  Replace frame-local entity stubs with shared study entities.

- name:

  Stable representation name.

## Value

Named representations, one representation, or representation IDs.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
study <- fmri_study(list(main = frame))
study_ids(study)
#> [1] "main"
study_frame(study, "main")
#> <fmri_frame> 3 observations x 4 features
#>   assays: bold 
#>   active: bold 
#>   space: volume_space ca8461c12469 
```
