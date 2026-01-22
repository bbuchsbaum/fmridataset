# Create Data Chunks for fmri_file_dataset Objects

This function creates data chunks for fmri_file_dataset objects. It
allows for the retrieval of run-wise or sequence-wise data chunks, as
well as arbitrary chunks.

## Usage

``` r
# S3 method for class 'fmri_file_dataset'
data_chunks(x, nchunks = 1, runwise = FALSE, ...)
```

## Arguments

- x:

  An object of class 'fmri_file_dataset'.

- nchunks:

  The number of data chunks to create. Default is 1.

- runwise:

  If TRUE, the data chunks are created run-wise. Default is FALSE.

- ...:

  Additional arguments.

## Value

A list of data chunks, with each chunk containing the data, voxel
indices, row indices, and chunk number.
