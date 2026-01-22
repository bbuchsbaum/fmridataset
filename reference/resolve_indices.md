# Resolve Indices from Series Selector

Converts a series selector specification into actual voxel indices
within the dataset mask.

## Usage

``` r
resolve_indices(selector, dataset, ...)
```

## Arguments

- selector:

  A series selector object (e.g., index_selector, voxel_selector)

- dataset:

  An fMRI dataset object providing spatial context

- ...:

  Additional arguments passed to methods

## Value

Integer vector of indices into the masked data

## Details

Series selectors provide various ways to specify spatial subsets of fMRI
data. This generic function resolves these specifications into actual
indices that can be used to extract data. Different selector types
support different selection methods:

- `index_selector`: Direct indices into masked data

- `voxel_selector`: 3D coordinates

- `roi_selector`: Region of interest masks

- `sphere_selector`: Spherical regions

## See also

[`series_selector`](https://bbuchsbaum.github.io/fmridataset/reference/series_selector.md)
for selector types,
[`fmri_series`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_series.md)
for using selectors to extract data

## Examples

``` r
# \donttest{
# Example with index selector
sel <- index_selector(1:10)
# indices <- resolve_indices(sel, dataset)

# Example with voxel coordinates
sel <- voxel_selector(cbind(x = 10, y = 20, z = 15))
# indices <- resolve_indices(sel, dataset)
# }
```
