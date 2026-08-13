# Convert a BIDS H5 Study Dataset to an fmri_group

Converts a `bids_h5_study_dataset` (or any `fmri_study_dataset`) to an
`fmri_group` object with one row per subject. Use this when you need
per-subject group operations via
[`group_map()`](https://bbuchsbaum.github.io/fmridataset/reference/group_map.md).

## Usage

``` r
study_to_group(x, ...)

# S3 method for class 'bids_h5_study_dataset'
study_to_group(x, ...)

# S3 method for class 'fmri_study_dataset'
study_to_group(x, ...)
```

## Arguments

- x:

  A `bids_h5_study_dataset` or `fmri_study_dataset`.

- ...:

  Currently unused.

## Value

An `fmri_group` with columns `subject_id` and `dataset`.
