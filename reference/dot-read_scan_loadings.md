# Read spatial loadings for a latent-mode scan

Reads the `[V, K]` loadings matrix stored at
`<scan_group_path>/data/loadings` in a BIDS HDF5 archive. Only
meaningful when `compression_mode = "latent"`.

## Usage

``` r
.read_scan_loadings(h5_handle, scan_group_path)
```

## Arguments

- h5_handle:

  An open
  [`hdf5r::H5File`](http://hhoeflin.github.io/hdf5r/reference/H5File-class.md)
  object.

- scan_group_path:

  Character string. HDF5 group path for the scan.

## Value

A numeric matrix of shape `[V, K]`, or `NULL` if the dataset is absent
(e.g. shared template mode where per-scan loadings are not stored).
