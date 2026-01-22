# Get Number of Timepoints from Sampling Frame

Extracts the total number of timepoints (volumes) across all runs from
objects containing temporal structure information.

## Usage

``` r
n_timepoints(x, ...)

# S3 method for class 'matrix_dataset'
n_timepoints(x, ...)

# S3 method for class 'fmri_dataset'
n_timepoints(x, ...)

# S3 method for class 'fmri_mem_dataset'
n_timepoints(x, ...)

# S3 method for class 'fmri_file_dataset'
n_timepoints(x, ...)

# S3 method for class 'fmri_study_dataset'
n_timepoints(x, ...)

# S3 method for class 'sampling_frame'
n_timepoints(x, ...)
```

## Arguments

- x:

  An object containing temporal structure (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Integer representing the total number of timepoints

## See also

[`n_runs`](https://bbuchsbaum.github.io/fmridataset/reference/n_runs.md)
for number of runs,
[`get_run_lengths`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_lengths.md)
for individual run lengths

## Examples

``` r
# \donttest{
# Create a sampling frame with 3 runs
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
n_timepoints(sf) # Returns: 330 (sum of run lengths)
#> [1] 330
# }
```
