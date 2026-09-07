# Construct a strictly aligned assay set

Construct a strictly aligned assay set

## Usage

``` r
aligned_assay_set(assays, observations, features)
```

## Arguments

- assays:

  Named matrices or `ArraySource` descriptors.

- observations:

  Observation axis.

- features:

  Feature axis.

## Value

An `aligned_assay_set`.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
obs <- axis_frame(data.frame(.obs_id = sprintf("vol-%d", 1:4)), axis = "observation")
feat <- feature_axis(feature_data(sp), space = sp)
bold <- matrix(rnorm(4 * n_features(sp)), nrow = 4)
aligned_assay_set(list(bold = bold), obs, feat)
#> $bold
#> $name
#> [1] "bold"
#> 
#> $source
#> $data
#>              [,1]       [,2]       [,3]       [,4]        [,5]       [,6]
#> [1,] -1.400043517  0.6215527 -0.2441996  2.0650249 -0.52201251  0.4681544
#> [2,]  0.255317055  1.1484116 -0.2827054 -1.6309894 -0.05260191  0.3629513
#> [3,] -2.437263611 -1.8218177 -0.5536994  0.5124269  0.54299634 -1.3045435
#> [4,] -0.005571287 -0.2473253  0.6289820 -1.8630115 -0.91407483  0.7377763
#>             [,7]       [,8]
#> [1,]  1.88850493 -0.8267890
#> [2,] -0.09744510 -1.5123997
#> [3,] -0.93584735  0.9353632
#> [4,] -0.01595031  0.1764886
#> 
#> $shape
#> [1] 4 8
#> 
#> $dtype
#> [1] "float64"
#> 
#> $chunks
#> [1] 4 8
#> 
#> $capabilities
#> [1] "row_slice"          "column_slice"       "block_slice"       
#> [4] "serializable"       "pushdown:all"       "pushdown:range"    
#> [7] "pushdown:positions"
#> 
#> $identity
#> [1] "object:46a6e935-95c3-44ef-8f32-656e4fe3d921"
#> 
#> $identity_basis
#> [1] "object"
#> 
#> $revision
#> NULL
#> 
#> $schema_version
#> [1] 1
#> 
#> $fingerprint
#> [1] "81c6706690e4e92df42548c93f2576f48db8fbf4e1d02e23ea02b0a5b00ccd40"
#> 
#> attr(,"class")
#> [1] "memory_source" "array_source" 
#> 
#> $dtype
#> [1] "float64"
#> 
#> $observation_digest
#> [1] "28df8f486f71cf37d9a5b1904e08bdc23d74ad6016c06fe1dda6720059b6f491"
#> 
#> $feature_digest
#> [1] "7da7a601714ec9063d1844682d3f9755e7b8785ad9d62d6280391d426c8a0c44"
#> 
#> $role
#> NULL
#> 
#> $units
#> NULL
#> 
#> $metadata
#> list()
#> 
#> attr(,"class")
#> [1] "aligned_assay"
#> 
#> attr(,"class")
#> [1] "aligned_assay_set" "list"             
```
