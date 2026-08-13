# Subset a BIDS H5 Study Dataset

Filters a `bids_h5_study_dataset` by task, subject, session, and/or run
using standard (non-NSE) evaluation. Returns a new
`bids_h5_study_dataset` built from the matching scans, sharing the same
underlying HDF5 file handle.

## Usage

``` r
subset_bids_h5(x, task = NULL, subject = NULL, session = NULL, run = NULL)
```

## Arguments

- x:

  A `bids_h5_study_dataset` object.

- task:

  Character vector of task names to keep, or `NULL` for all.

- subject:

  Character vector of subject IDs to keep, or `NULL` for all.

- session:

  Character vector of session names to keep, or `NULL` for all.

- run:

  Character vector of BIDS run labels to keep, or `NULL` for all.

## Value

A new `bids_h5_study_dataset` containing only the matching scans.
