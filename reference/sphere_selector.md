# Spherical ROI Series Selector

Select voxels within a spherical region.

## Usage

``` r
sphere_selector(center, radius)
```

## Arguments

- center:

  Numeric vector of length 3 (x, y, z) specifying sphere center

- radius:

  Numeric radius in voxel units

## Value

An object of class `sphere_selector`

## Examples

``` r
# Select 10-voxel radius sphere around voxel (30, 30, 20)
sel <- sphere_selector(center = c(30, 30, 20), radius = 10)
```
