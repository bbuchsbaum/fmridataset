# Get Scan Manifest from a BIDS H5 Dataset

Generic function to extract the per-scan metadata table from a BIDS HDF5
study dataset.

## Usage

``` r
scan_manifest(x, ...)

# S3 method for class 'bids_h5_study_dataset'
scan_manifest(x, ...)
```

## Arguments

- x:

  A BIDS H5 study dataset object

- ...:

  Additional arguments passed to methods

## Value

A tibble with columns: scan_name, subject, task, session, run, n_time,
has_events, has_confounds
