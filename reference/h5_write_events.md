# Write Events to HDF5 Scan Group

Writes a data.frame (events table) to the `events/` subgroup of a scan
HDF5 group, using one dataset per column. Stores the `n_events`
attribute on the group.

## Usage

``` r
h5_write_events(h5_group, events, compression = 4L)
```

## Arguments

- h5_group:

  An
  [`hdf5r::H5Group`](http://hhoeflin.github.io/hdf5r/reference/H5Group-class.md)
  object for the scan (e.g.
  `h5file[["scans/sub-01_task-nback_run-01"]]`).

- events:

  A data.frame with event columns (onset, duration, trial_type, ...).
  Must have at least `onset` and `duration`.

- compression:

  Integer 0-9. HDF5 gzip compression level (default 4).

## Value

Invisible NULL.
