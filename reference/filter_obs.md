# Filter frame observations using scalar metadata

Filter frame observations using scalar metadata

## Usage

``` r
filter_obs(x, predicate, resolve = TRUE)
```

## Arguments

- x:

  An `fmri_frame` or view.

- predicate:

  A metadata expression returning one logical value per observation.

- resolve:

  Whether the predicate may use namespaced entity metadata.

## Value

An `fmri_view`.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(
    .obs_id = sprintf("vol-%d", 1:4),
    run_id = rep(c("run-1", "run-2"), each = 2)
  ),
  space = sp
)
filter_obs(frame, run_id == "run-1")
#> <fmri_view> 2 observations x 8 features
#>   base: 4 x 8 
#>   assays: bold 
```
