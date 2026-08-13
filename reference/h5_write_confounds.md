# Write Confound Matrix to HDF5 Scan Group

Writes a confound regressor matrix to `confounds/data` in the scan
group, storing column names as a `names` attribute.

## Usage

``` r
h5_write_confounds(h5_group, confounds, compression = 4L)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  for the scan.

- confounds:

  A matrix or data.frame `[T, n_confounds]`. If `NULL`, nothing is
  written.

- compression:

  Integer 0-9. HDF5 gzip compression level (default 4).

## Value

Invisible NULL.
