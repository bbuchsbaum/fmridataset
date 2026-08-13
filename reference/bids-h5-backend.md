# BIDS H5 Scan Backend

A lightweight storage backend for a single scan stored inside a shared
BIDS HDF5 archive. Many `bids_h5_scan_backend` objects share a single
`h5_shared_connection`, allowing one file handle to serve an entire
study without leaking file descriptors.

## Details

The backend operates in **feature-space**: columns are K features
(parcels for `compression_mode = "parcellated"`, or latent components
for `compression_mode = "latent"`), not V voxels. This satisfies the
backend contract by reporting `spatial = c(K, 1, 1)` and
`mask = rep(TRUE, K)`, so
[`validate_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_backend.md)
passes and all downstream components (study_backend, as_delarr,
data_chunks) work unchanged.

For `"parcellated"` mode, data is read from
`/scans/<name>/data/summary_data` (shape `[T, K]`). For `"latent"` mode,
the temporal basis is read from `/scans/<name>/data/basis` (shape
`[T, K]`). Loadings and offset can be accessed via
[`.read_scan_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/dot-read_scan_loadings.md)
and
[`.read_scan_offset()`](https://bbuchsbaum.github.io/fmridataset/reference/dot-read_scan_offset.md)
helpers.

Original voxel geometry is stored in the HDF5 file under `/spatial/` and
(for parcellated mode) `/parcellation/`, but does **not** flow through
the backend contract.
