# Write Censor Vector to HDF5 Scan Group

Writes a logical or integer censor vector as a `uint8` dataset named
`censor` in the scan group. Values: 0 = keep, 1 = censor.

## Usage

``` r
h5_write_censor(h5_group, censor, compression = 4L)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  for the scan.

- censor:

  Integer or logical vector of length T. `NULL` means no censor vector
  is written (all timepoints kept).

- compression:

  Integer 0-9 (default 4).

## Value

Invisible NULL.
