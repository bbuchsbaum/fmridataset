# Create an Execution Strategy for Data Processing

This function creates an execution strategy that can be used to process
fMRI datasets in different ways: voxelwise, runwise, or chunkwise.

## Usage

``` r
exec_strategy(
  strategy = c("voxelwise", "runwise", "chunkwise"),
  nchunks = NULL
)
```

## Arguments

- strategy:

  Character string specifying the processing strategy. Options are
  "voxelwise", "runwise", or "chunkwise".

- nchunks:

  Number of chunks to use for "chunkwise" strategy. Ignored for other
  strategies.

## Value

A function that takes a dataset and returns a chunk iterator configured
according to the specified strategy.
