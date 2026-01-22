# Create Data Chunks for fmri_mem_dataset Objects

This function creates data chunks for fmri_mem_dataset objects. It
allows for the retrieval of run-wise or sequence-wise data chunks, as
well as arbitrary chunks.

## Usage

``` r
# S3 method for class 'fmri_mem_dataset'
data_chunks(x, nchunks = 1, runwise = FALSE, ...)
```

## Arguments

- x:

  An object of class 'fmri_mem_dataset'.

- nchunks:

  The number of data chunks to create. Default is 1.

- runwise:

  If TRUE, the data chunks are created run-wise. Default is FALSE.

- ...:

  Additional arguments.

## Value

A list of data chunks, with each chunk containing the data, voxel
indices, row indices, and chunk number.

## Examples

``` r
if (FALSE) { # \dontrun{
# Create a simple fmri_mem_dataset for demonstration
d <- c(10, 10, 10, 10)
nvec <- neuroim2::NeuroVec(array(rnorm(prod(d)), d), space = neuroim2::NeuroSpace(d))
mask <- neuroim2::LogicalNeuroVol(array(TRUE, d[1:3]), neuroim2::NeuroSpace(d[1:3]))
dset <- fmri_mem_dataset(list(nvec), mask, TR = 2)

# Create an iterator with 5 chunks
iter <- data_chunks(dset, nchunks = 5)
`%do%` <- foreach::`%do%`
y <- foreach::foreach(chunk = iter) %do% {
  colMeans(chunk$data)
}
length(y) == 5

# Create an iterator with 100 chunks
iter <- data_chunks(dset, nchunks = 100)
y <- foreach::foreach(chunk = iter) %do% {
  colMeans(chunk$data)
}
length(y) == 100

# Create a "runwise" iterator
iter <- data_chunks(dset, runwise = TRUE)
y <- foreach::foreach(chunk = iter) %do% {
  colMeans(chunk$data)
}
length(y) == 1
} # }
```
