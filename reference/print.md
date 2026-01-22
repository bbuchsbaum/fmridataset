# Print Methods for fmridataset Objects

Display formatted summaries of fmridataset objects including datasets,
chunk iterators, and data chunks.

This function prints a summary of a chunk iterator.

This function prints a summary of a data chunk.

## Usage

``` r
# S3 method for class 'fmri_dataset'
print(x, full = FALSE, ...)

# S3 method for class 'fmri_dataset'
summary(object, ...)

# S3 method for class 'chunkiter'
print(x, ...)

# S3 method for class 'data_chunk'
print(x, ...)

# S3 method for class 'matrix_dataset'
print(x, ...)
```

## Arguments

- x:

  A data_chunk object.

- full:

  Logical; if TRUE, print additional details for datasets (default:
  FALSE)

- ...:

  Additional arguments (ignored).

- object:

  An object to summarize (for summary methods)

## Value

The object invisibly

## Examples

``` r
# \donttest{
# Print dataset summary
# dataset <- fmri_dataset(...)
# print(dataset)
# print(dataset, full = TRUE)  # More details
# }
```
