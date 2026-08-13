# Build per-subject fmri_dataset from a set of scan rows and scan backends

Build per-subject fmri_dataset from a set of scan rows and scan backends

## Usage

``` r
.build_subject_dataset(scan_rows, scan_backends, h5, tr, subject_id)
```

## Arguments

- scan_rows:

  Rows of scan_manifest for this subject (tibble).

- scan_backends:

  Named list of bids_h5_scan_backend, keyed by scan_name.

- h5:

  The open H5File handle (for reading events/censor).

- tr:

  Numeric TR in seconds.

- subject_id:

  Character subject ID for event_table annotation.
