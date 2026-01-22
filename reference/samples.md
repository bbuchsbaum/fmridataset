# Get Sample Indices from Sampling Frame

Generates a vector of timepoint indices, typically used for time series
analysis or indexing operations.

## Usage

``` r
samples(x, ...)

# S3 method for class 'matrix_dataset'
samples(x, ...)

# S3 method for class 'fmri_dataset'
samples(x, ...)

# S3 method for class 'fmri_mem_dataset'
samples(x, ...)

# S3 method for class 'fmri_file_dataset'
samples(x, ...)

# S3 method for class 'fmri_study_dataset'
samples(x, ...)

# S3 method for class 'sampling_frame'
samples(x, ...)
```

## Arguments

- x:

  An object containing temporal structure (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Integer vector from 1 to the total number of timepoints

## See also

[`n_timepoints`](https://bbuchsbaum.github.io/fmridataset/reference/n_timepoints.md)
for total number of samples,
[`blockids`](https://bbuchsbaum.github.io/fmridataset/reference/blockids.md)
for run membership

## Examples

``` r
# \donttest{
# Create a sampling frame
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
s <- samples(sf)
length(s) # 220
#> [1] 220
range(s) # c(1, 220)
#> [1]   1 220
# }
```
