# Construct a packed volumetric feature space

Construct a packed volumetric feature space

## Usage

``` r
volume_space(
  dim,
  affine = diag(4),
  support = NULL,
  template = NULL,
  units = "mm",
  metadata = list()
)
```

## Arguments

- dim:

  Three spatial dimensions.

- affine:

  A 4 by 4 voxel-to-world affine.

- support:

  Logical full-volume support or packed linear indices.

- template:

  Optional template/native-space identity.

- units:

  Spatial units.

- metadata:

  Additional serializable metadata.

## Value

A `volume_space`.

## Examples

``` r
volume_space(c(2, 2, 2), affine = diag(4), support = 1:4)
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
#> [1] 1 2 3 4
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
