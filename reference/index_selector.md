# Index-based Series Selector

Select voxels by their direct indices in the masked data.

## Usage

``` r
index_selector(indices)
```

## Arguments

- indices:

  Integer vector of voxel indices

## Value

An object of class `index_selector`

## Examples

``` r
# Select first 10 voxels
sel <- index_selector(1:10)

# Select specific voxels
sel <- index_selector(c(1, 5, 10, 20))
```
