# Read Censor Vector from HDF5 Scan Group

Reads the `censor` dataset from a scan group. Returns `NULL` if absent
(implying all timepoints are kept).

## Usage

``` r
h5_read_censor(h5_group)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  for the scan.

## Value

Integer vector (0/1) of length T, or `NULL`.
