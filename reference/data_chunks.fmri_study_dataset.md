# Create Data Chunks for fmri_study_dataset Objects

This function creates data chunks for multi-subject study datasets.

## Usage

``` r
# S3 method for class 'fmri_study_dataset'
data_chunks(x, nchunks = 1, runwise = FALSE, ...)
```

## Arguments

- x:

  An object of class 'fmri_study_dataset'

- nchunks:

  The number of chunks to split the data into. Default is 1.

- runwise:

  If TRUE, creates run-wise chunks instead of arbitrary chunks

- ...:

  Additional arguments passed to methods

## Value

A list of data chunks, each containing data, indices and chunk number
