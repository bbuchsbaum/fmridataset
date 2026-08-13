# Get Participant IDs from a Dataset

Generic function to extract participant/subject identifiers from
study-level fMRI dataset objects. Defined here in fmridataset so methods
work without requiring bidser to be installed.

## Usage

``` r
participants(x, ...)

# S3 method for class 'bids_h5_study_dataset'
participants(x, ...)
```

## Arguments

- x:

  A study dataset object (e.g., bids_h5_study_dataset)

- ...:

  Additional arguments passed to methods

## Value

Character vector of participant identifiers
