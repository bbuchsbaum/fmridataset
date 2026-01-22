# Voxel Coordinate Series Selector

Select voxels by their 3D coordinates in the image space.

## Usage

``` r
voxel_selector(coords)
```

## Arguments

- coords:

  Matrix with 3 columns (x, y, z) or vector of length 3

## Value

An object of class `voxel_selector`

## Examples

``` r
# Select single voxel
sel <- voxel_selector(c(10, 20, 15))

# Select multiple voxels
coords <- cbind(x = c(10, 20), y = c(20, 30), z = c(15, 15))
sel <- voxel_selector(coords)
```
