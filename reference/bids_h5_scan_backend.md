# Create a BIDS H5 Scan Backend

Constructs a lightweight backend for one scan stored in a BIDS HDF5
archive. The backend holds a reference to a shared
`h5_shared_connection` and reads data from the HDF5 group for this scan.
For `"parcellated"` mode data is at `/scans/<name>/data/summary_data`;
for `"latent"` mode the temporal basis is at `/scans/<name>/data/basis`.

## Usage

``` r
bids_h5_scan_backend(
  h5_connection,
  scan_group_path,
  n_features,
  n_time,
  metadata = list(),
  compression_mode = "parcellated"
)
```

## Arguments

- h5_connection:

  An `h5_shared_connection` object (shared across scans).

- scan_group_path:

  Character string. HDF5 group path for this scan, e.g.
  `"/scans/sub-01_task-nback_run-01"`.

- n_features:

  Integer. Number of features (parcels or latent components) — i.e. the
  number of columns in the data matrix.

- n_time:

  Integer. Number of timepoints (rows in the data matrix).

- metadata:

  Named list of scan metadata (subject, task, session, run, tr). May be
  empty; defaults to [`list()`](https://rdrr.io/r/base/list.html).

- compression_mode:

  Character. Either `"parcellated"` (default) or `"latent"`. Determines
  which HDF5 dataset is read by `backend_get_data`.

## Value

A `bids_h5_scan_backend` / `storage_backend` environment.
