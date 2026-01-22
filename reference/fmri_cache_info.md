# Get cache information and statistics

Returns information about the current state of the fmridataset cache,
including size, number of objects, hit/miss rates, and memory usage.

## Usage

``` r
fmri_cache_info()
```

## Value

Named list with cache statistics

## Examples

``` r
if (FALSE) { # \dontrun{
# Get cache information
cache_info <- fmri_cache_info()
print(cache_info)
} # }
```
