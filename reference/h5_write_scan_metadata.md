# Write Per-Scan Metadata to HDF5

Writes scalar metadata fields (subject, task, session, run, tr) as
individual string/numeric datasets under `metadata/` in a scan group.

## Usage

``` r
h5_write_scan_metadata(h5_group, meta)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  for the scan.

- meta:

  Named list with scalar elements: subject, task, session (may be NULL),
  run, tr.

## Value

Invisible NULL.
