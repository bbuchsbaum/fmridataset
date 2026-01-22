# Create Data Chunks for Processing

Generic function to create data chunks for parallel processing from
various fMRI dataset types. Supports different chunking strategies.

## Usage

``` r
data_chunks(x, nchunks = 1, runwise = FALSE, ...)
```

## Arguments

- x:

  An fMRI dataset object

- nchunks:

  Number of chunks to create (default: 1)

- runwise:

  If TRUE, create run-wise chunks (default: FALSE)

- ...:

  Additional arguments passed to methods

## Value

A chunk iterator object that yields data chunks when iterated

## Details

Large fMRI datasets can be processed more efficiently by dividing them
into chunks. This function creates an iterator that yields data chunks
for parallel or sequential processing. Two chunking strategies are
supported:

- Equal-sized chunks: Divides voxels into approximately equal groups

- Run-wise chunks: Each chunk contains all voxels from one or more
  complete runs

## See also

[`iter`](https://rdrr.io/pkg/iterators/man/iter.html) for iteration
concepts

## Examples

``` r
# \donttest{
# Create a dataset
mat <- matrix(rnorm(100 * 1000), nrow = 100, ncol = 1000)
ds <- matrix_dataset(mat, TR = 2, run_length = 100)

# Create 4 chunks for parallel processing
chunks <- data_chunks(ds, nchunks = 4)

# Process chunks (example)
# results <- foreach(chunk = chunks) %dopar% {
#   process_chunk(chunk)
# }
# }
```
