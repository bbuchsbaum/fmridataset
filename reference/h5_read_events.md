# Read Events from HDF5 Scan Group

Reads the column-array events stored under `events/` in a scan group and
reassembles them into a data.frame. Returns `NULL` if the group has no
`events/` subgroup or the group is empty.

## Usage

``` r
h5_read_events(h5_group)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  for the scan.

## Value

A data.frame of events, or `NULL`.
