# Read Confound Matrix from HDF5 Scan Group

Reads the confound matrix stored under `confounds/data` and returns a
data.frame with the original column names restored from the `names`
attribute. Returns `NULL` if no confounds are stored.

## Usage

``` r
h5_read_confounds(h5_group)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  for the scan.

## Value

A data.frame `[T, n_confounds]`, or `NULL`.
