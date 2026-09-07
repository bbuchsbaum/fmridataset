# Recover the spatial domain of a NIfTI source

Recover the spatial domain of a NIfTI source

## Usage

``` r
nifti_source_space(x, template = NULL)
```

## Arguments

- x:

  A `nifti_array_source`.

- template:

  Optional template or native-space label.

## Value

A compatible `volume_space`.

## Examples

``` r
path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
if (nzchar(path)) {
  src <- nifti_array_source(path, path)
  spatial <- nifti_source_space(src, template = "fixture")
  n_features(spatial)
}
#> [1] 29532
```
