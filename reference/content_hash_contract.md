# Content-hash contract

Content hashing version 1 is a chained SHA-256 over fixed-size leaves of
the realized values in row-major order. The digest depends on the shape,
the realized R mode, and every value, and on nothing else: not on the
storage dtype, the chunk grid, the read block size, or the composition
of sources that produced the values.

## Usage

``` r
content_hash_contract()
```

## Value

A serializable content-hash contract descriptor.

## Examples

``` r
content_hash_contract()
#> $id
#> [1] "org.fmridataset.content-hash/v1"
#> 
#> $version
#> [1] 1
#> 
#> $algorithm
#> [1] "sha256"
#> 
#> $order
#> [1] "row-major"
#> 
#> $leaf_values
#> [1] 65536
#> 
#> $byte_order
#> [1] "big-endian"
#> 
#> $nan
#> [1] "canonical-payload"
#> 
#> $negative_zero
#> [1] "preserved"
#> 
#> $portability
#> [1] "R-only"
#> 
```
