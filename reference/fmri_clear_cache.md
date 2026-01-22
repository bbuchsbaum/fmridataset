# Clear fmridataset cache

Clears the internal cache used by fmridataset for memoized file
operations. This can be useful to free memory or force re-reading of
files.

## Usage

``` r
fmri_clear_cache()
```

## Value

NULL (invisibly)

## Examples

``` r
if (FALSE) { # \dontrun{
# Clear the cache to free memory
fmri_clear_cache()
} # }
```
