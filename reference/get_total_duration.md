# Get Total Duration from Sampling Frame

Calculates the total duration of the fMRI acquisition in seconds across
all runs.

## Usage

``` r
get_total_duration(x, ...)

# S3 method for class 'matrix_dataset'
get_total_duration(x, ...)

# S3 method for class 'fmri_dataset'
get_total_duration(x, ...)

# S3 method for class 'fmri_mem_dataset'
get_total_duration(x, ...)

# S3 method for class 'fmri_file_dataset'
get_total_duration(x, ...)

# S3 method for class 'fmri_study_dataset'
get_total_duration(x, ...)

# S3 method for class 'sampling_frame'
get_total_duration(x, ...)
```

## Arguments

- x:

  An object containing temporal structure (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Numeric value representing total duration in seconds

## See also

[`get_run_duration`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_duration.md)
for individual run durations,
[`get_TR`](https://bbuchsbaum.github.io/fmridataset/reference/get_TR.md)
for repetition time

## Examples

``` r
# \donttest{
# Create a sampling frame: 220 timepoints with TR = 2 seconds
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
get_total_duration(sf) # Returns: 440 seconds
#> [1] 440
# }
```
