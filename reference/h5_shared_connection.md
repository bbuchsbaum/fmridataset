# Create a Shared H5 Connection

A ref-counted wrapper around an
[`hdf5r::H5File`](http://hhoeflin.github.io/hdf5r/reference/H5File-class.md)
object. Multiple `bids_h5_scan_backend` objects share one connection;
the file is closed only when the last backend releases it.

## Usage

``` r
h5_shared_connection(file)
```

## Arguments

- file:

  Character string. Path to the HDF5 file to open.

## Value

An environment of class `h5_shared_connection` with fields:

- `file`: the file path

- `handle`: the open
  [`hdf5r::H5File`](http://hhoeflin.github.io/hdf5r/reference/H5File-class.md)
  object

- `ref_count`: integer, number of live backends holding this connection

And methods `acquire()` and `release()`.
