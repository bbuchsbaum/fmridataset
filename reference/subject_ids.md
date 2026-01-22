# Get Subject IDs from Multi-Subject Dataset

Generic function to extract subject identifiers from multi-subject fMRI
dataset objects.

## Usage

``` r
subject_ids(x, ...)

# S3 method for class 'fmri_study_dataset'
subject_ids(x, ...)
```

## Arguments

- x:

  A multi-subject dataset object (e.g., fmri_study_dataset)

- ...:

  Additional arguments passed to methods

## Value

Character vector of subject identifiers

## Details

Multi-subject datasets contain data from multiple participants. This
function extracts the subject identifiers associated with each dataset.
The order of subject IDs corresponds to the order of datasets.
