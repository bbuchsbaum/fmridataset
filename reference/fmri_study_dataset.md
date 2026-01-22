# Create an fmri_study_dataset

High level constructor that combines multiple `fmri_dataset` objects
into a single study-level dataset using `study_backend`.

## Usage

``` r
fmri_study_dataset(datasets, subject_ids = NULL)
```

## Arguments

- datasets:

  A list of `fmri_dataset` objects

- subject_ids:

  Optional vector of subject identifiers

## Value

An object of class `fmri_study_dataset`
