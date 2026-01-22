# Mask-based Series Selector

Select voxels that are TRUE in a binary mask.

## Usage

``` r
mask_selector(mask)
```

## Arguments

- mask:

  A logical vector matching the dataset's mask length, or a 3D logical
  array

## Value

An object of class `mask_selector`

## Examples

``` r
if (FALSE) { # \dontrun{
# Using a logical vector
mask_vec <- backend_get_mask(dataset$backend)
sel <- mask_selector(mask_vec > 0.5)
} # }
```
