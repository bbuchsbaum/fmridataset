# Resolve Spatial Selector

Resolve Spatial Selector

## Usage

``` r
resolve_selector(dataset, selector)
```

## Arguments

- dataset:

  An `fmri_dataset` object.

- selector:

  Spatial selector or `NULL` for all voxels. Supported types are integer
  indices, coordinate matrices with three columns, and logical or ROI
  volumes.

## Value

Integer vector of voxel indices within the dataset mask.
