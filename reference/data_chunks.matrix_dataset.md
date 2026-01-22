# Create Data Chunks for matrix_dataset Objects

This function creates data chunks for matrix_dataset objects. It allows
for the retrieval of run-wise or sequence-wise data chunks, as well as
arbitrary chunks.

## Usage

``` r
# S3 method for class 'matrix_dataset'
data_chunks(x, nchunks = 1, runwise = FALSE, ...)
```

## Arguments

- x:

  An object of class 'matrix_dataset'

- nchunks:

  The number of chunks to split the data into. Default is 1.

- runwise:

  If TRUE, creates run-wise chunks instead of arbitrary chunks

- ...:

  Additional arguments passed to methods

## Value

A list of data chunks, each containing data, indices and chunk number
