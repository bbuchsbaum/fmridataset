# Get Block IDs from Sampling Frame

Generates a vector of block/run identifiers for each timepoint.

## Usage

``` r
blockids(x, ...)

# S3 method for class 'matrix_dataset'
blockids(x, ...)

# S3 method for class 'fmri_dataset'
blockids(x, ...)

# S3 method for class 'fmri_mem_dataset'
blockids(x, ...)

# S3 method for class 'fmri_file_dataset'
blockids(x, ...)

# S3 method for class 'fmri_study_dataset'
blockids(x, ...)

# S3 method for class 'sampling_frame'
blockids(x, ...)
```

## Arguments

- x:

  An object containing temporal structure (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Integer vector of length equal to total timepoints, with values
indicating run membership (1 for first run, 2 for second, etc.)

## Details

This function creates a vector where each element indicates which
run/block the corresponding timepoint belongs to. This is useful for
run-wise analyses or modeling run effects.

## See also

[`get_run_lengths`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_lengths.md)
for run lengths,
[`samples`](https://bbuchsbaum.github.io/fmridataset/reference/samples.md)
for timepoint indices

## Examples

``` r
# \donttest{
# Create a sampling frame with 2 runs of different lengths
sf <- fmrihrf::sampling_frame(blocklens = c(3, 4), TR = 2)
blockids(sf) # Returns: c(1, 1, 1, 2, 2, 2, 2)
#> [1] 1 1 1 2 2 2 2
# }
```
