# Get Run Duration from Sampling Frame

Calculates the duration of each run in seconds.

## Usage

``` r
get_run_duration(x, ...)

# S3 method for class 'matrix_dataset'
get_run_duration(x, ...)

# S3 method for class 'fmri_dataset'
get_run_duration(x, ...)

# S3 method for class 'fmri_mem_dataset'
get_run_duration(x, ...)

# S3 method for class 'fmri_file_dataset'
get_run_duration(x, ...)

# S3 method for class 'fmri_study_dataset'
get_run_duration(x, ...)

# S3 method for class 'sampling_frame'
get_run_duration(x, ...)
```

## Arguments

- x:

  An object containing temporal structure (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Numeric vector where each element represents the duration of the
corresponding run in seconds

## See also

[`get_total_duration`](https://bbuchsbaum.github.io/fmridataset/reference/get_total_duration.md)
for total duration,
[`get_run_lengths`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_lengths.md)
for run lengths in timepoints

## Examples

``` r
# \donttest{
# Create a sampling frame with different run lengths
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
get_run_duration(sf) # Returns: c(200, 240) seconds
#> [1] 200 240
# }
```
