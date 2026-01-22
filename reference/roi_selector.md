# ROI-based Series Selector

Select voxels within a region of interest (ROI) volume or mask.

## Usage

``` r
roi_selector(roi)
```

## Arguments

- roi:

  A 3D array, ROIVol, LogicalNeuroVol, or similar mask object

## Value

An object of class `roi_selector`

## Examples

``` r
if (FALSE) { # \dontrun{
# Using a binary mask
mask <- array(FALSE, dim = c(64, 64, 30))
mask[30:40, 30:40, 15:20] <- TRUE
sel <- roi_selector(mask)
} # }
```
