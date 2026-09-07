# Collect one frame assay under an explicit memory budget

Collect one frame assay under an explicit memory budget

## Usage

``` r
collect_assay(
  x,
  assay = active_assay(x),
  memory_budget = getOption("fmridataset.collect_budget", 2 * 1024^3),
  force = FALSE
)
```

## Arguments

- x:

  An `fmri_frame` or view.

- assay:

  Assay name.

- memory_budget:

  Maximum estimated peak bytes, including the retained output and source
  conversion or decompression buffers.

- force:

  Allow collection above the estimated peak budget.

## Value

A dense matrix.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
collect_assay(frame)
#>            [,1]        [,2]        [,3]       [,4]       [,5]       [,6]
#> [1,] -0.5282641  0.08171963 -0.85520250 -0.1626763  0.9799567 -1.5090998
#> [2,]  0.1921494 -1.30511701 -0.28689522 -0.8273102  1.3217810  1.5327415
#> [3,] -1.1461997 -0.94491206  0.89496163  1.8765056 -1.1197108  0.4291474
#> [4,]  0.8461847  0.45434159  0.06730444  0.7664402  0.5145998  0.1221034
#>            [,7]        [,8]
#> [1,] -1.1380124  0.03849955
#> [2,] -0.5580151 -0.35638119
#> [3,]  1.0525385  0.78284410
#> [4,]  0.6776836  0.80441162
```
