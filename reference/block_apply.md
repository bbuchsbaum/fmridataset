# Apply a function to bounded feature blocks

Apply a function to bounded feature blocks

## Usage

``` r
block_apply(x, FUN, block_size = 4096L, assay = active_assay(x), ...)
```

## Arguments

- x:

  An `fmri_frame` or view.

- FUN:

  Function receiving an observation-by-feature matrix and feature IDs.

- block_size:

  Number of features per block.

- assay:

  Assay name.

- ...:

  Additional arguments passed to `FUN`.

## Value

A list of block results.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
block_apply(frame, function(mat, ids) colMeans(mat), block_size = 4L)
#> [[1]]
#> [1]  0.4030433 -0.7562311 -0.8855047 -0.1006778
#> 
#> [[2]]
#> [1] 0.62804704 0.01570424 1.07680091 0.11770979
#> 
```
