# Compose scan backends + manifest into a bids_h5_study_dataset

Shared by bids_h5_dataset() and subset_bids_h5(). Takes the full
manifest and a flat named list of scan_backends (keyed by scan_name),
builds the subject-level datasets, and returns a bids_h5_study_dataset.

## Usage

``` r
.compose_bids_h5_study_dataset(
  manifest,
  scan_backends,
  h5,
  h5_connection,
  tr,
  bids_meta,
  compression_mode = "parcellated"
)
```

## Arguments

- manifest:

  Tibble — the scan manifest (subset of the full one).

- scan_backends:

  Named list of bids_h5_scan_backend objects.

- h5:

  The open hdf5r H5File handle.

- h5_connection:

  The h5_shared_connection (stored on the result).

- tr:

  Numeric TR.

- bids_meta:

  Named list: space, pipeline, name (from /bids/).

- compression_mode:

  Character. Either "parcellated" or "latent".
