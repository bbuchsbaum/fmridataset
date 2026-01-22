# Resize the fmridataset cache

Changes the maximum size of the cache. This will immediately evict
objects if the new size is smaller than the current cache contents.

## Usage

``` r
fmri_cache_resize(size_mb)
```

## Arguments

- size_mb:

  Numeric cache size in megabytes

## Value

NULL (invisibly)

## Examples

``` r
if (FALSE) { # \dontrun{
# Resize cache to 1GB
fmri_cache_resize(1024)

# Check new size
fmri_cache_info()
} # }
```
