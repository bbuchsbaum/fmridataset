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
