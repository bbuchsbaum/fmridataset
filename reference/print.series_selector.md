# Print Methods for Series Selectors

Display formatted summaries of series selector objects.

## Usage

``` r
# S3 method for class 'series_selector'
print(x, ...)
```

## Arguments

- x:

  A series selector object

- ...:

  Additional arguments (currently unused)

## Value

The object invisibly

## Examples

``` r
# Print different selector types
sel1 <- index_selector(1:10)
print(sel1)
#> <index_selector>
#>   indices: 1, 2, 3, ... (10 total)

sel2 <- voxel_selector(c(10, 20, 15))
print(sel2)
#> <voxel_selector>
#>   coordinates: 1 voxel(s)
#>     [10, 20, 15]
```
