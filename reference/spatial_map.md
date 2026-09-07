# Recover a spatial map for one observation

Recover a spatial map for one observation

## Usage

``` r
spatial_map(x, observation, assay = active_assay(x))
```

## Arguments

- x:

  An `fmri_frame` or view.

- observation:

  Observation ID or one integer position.

- assay:

  Assay name.

## Value

A reconstructed spatial object.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
spatial_map(frame, observation = 1L)
#> <DenseNeuroVol> [6.9 Kb] 
#> ── Spatial ───────────────────────────────────────────────────────────────────── 
#>   Dimensions    : 2 x 2 x 2
#>   Spacing       : 1 x 1 x 1 mm
#>   Origin        : 0, 0, 0
#>   Orientation   : RAS
#> ── Data ──────────────────────────────────────────────────────────────────────── 
#>   Range         : [-1.489, 1.336]
```
