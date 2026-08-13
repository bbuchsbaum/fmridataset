# Read per-voxel offset for a latent-mode scan

Reads the `[V]` offset vector stored at `<scan_group_path>/data/offset`
in a BIDS HDF5 archive. Returns `numeric(0)` if the dataset is absent
(offset not stored). Only meaningful when `compression_mode = "latent"`.

## Usage

``` r
.read_scan_offset(h5_handle, scan_group_path)
```

## Arguments

- h5_handle:

  An open
  [`hdf5r::H5File`](http://hhoeflin.github.io/hdf5r/reference/H5File-class.md)
  object.

- scan_group_path:

  Character string. HDF5 group path for the scan.

## Value

A numeric vector of length V, or `numeric(0)`.
