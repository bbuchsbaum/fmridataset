# Get Confound Regressors from a BIDS H5 Dataset

Generic function to retrieve confound regressor matrices stored in a
BIDS HDF5 dataset.

## Usage

``` r
get_confounds(x, ...)

# S3 method for class 'bids_h5_study_dataset'
get_confounds(x, scan_name = NULL, subject = NULL, task = NULL, ...)
```

## Arguments

- x:

  A BIDS H5 study dataset object

- ...:

  Additional arguments passed to methods

- scan_name:

  Character. Scan name key (exact match), or `NULL`.

- subject:

  Character. Subject ID filter, or `NULL`.

- task:

  Character. Task filter, or `NULL`.

## Value

A tibble (single scan) or named list of tibbles (multiple scans)
