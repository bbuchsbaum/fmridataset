# FDS logical schema identity

FDS version 1 is the backend-neutral semantic contract for persisted
`fmri_frame` objects. Physical codecs may add locations, chunks,
compression, and checksums outside this manifest, but cannot change its
field meanings.

## Usage

``` r
fds_schema()

fds_schema_version()
```

## Value

`fds_schema()` returns the immutable schema identity;
`fds_schema_version()` returns its integer major version.

## Examples

``` r
fds_schema()
#> $id
#> [1] "org.fmridataset.fds/v1"
#> 
#> $version
#> [1] 1
#> 
#> $object_types
#> [1] "fmri_frame"
#> 
fds_schema_version()
#> [1] 1
```
