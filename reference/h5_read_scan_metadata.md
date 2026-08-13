# Read Per-Scan Metadata from HDF5

Reads the scalar metadata datasets from `metadata/` in a scan group.

## Usage

``` r
h5_read_scan_metadata(h5_group)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  for the scan.

## Value

Named list of metadata values.
