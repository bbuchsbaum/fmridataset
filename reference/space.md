# Feature-space accessor

Feature-space accessor

## Usage

``` r
space(x, ...)
```

## Arguments

- x:

  An object with spatial identity.

- ...:

  Additional arguments.

## Value

A `FeatureSpace`.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
space(frame)
#> $dim
#> [1] 2 2 2
#> 
#> $affine
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
#> 
#> $support
#> [1] 1 2 3 4 5 6 7 8
#> 
#> $template
#> NULL
#> 
#> $units
#> [1] "mm"
#> 
#> $metadata
#> list()
#> 
#> $schema_version
#> [1] 1
#> 
#> attr(,"class")
#> [1] "volume_space"  "feature_space"
```
