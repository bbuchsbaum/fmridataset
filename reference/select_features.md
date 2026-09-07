# Select frame features using feature metadata

Select frame features using feature metadata

## Usage

``` r
select_features(x, predicate)
```

## Arguments

- x:

  An `fmri_frame` or view.

- predicate:

  A metadata expression returning one logical value per feature.

## Value

An `fmri_view`.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
select_features(frame, i == 1)
#> <fmri_view> 4 observations x 4 features
#>   base: 4 x 8 
#>   assays: bold 
```
