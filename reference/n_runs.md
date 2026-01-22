# Get Number of Runs from Sampling Frame

Extracts the total number of runs/blocks from objects containing
temporal structure information.

## Usage

``` r
n_runs(x, ...)

# S3 method for class 'matrix_dataset'
n_runs(x, ...)

# S3 method for class 'fmri_dataset'
n_runs(x, ...)

# S3 method for class 'fmri_mem_dataset'
n_runs(x, ...)

# S3 method for class 'fmri_file_dataset'
n_runs(x, ...)

# S3 method for class 'fmri_study_dataset'
n_runs(x, ...)

# S3 method for class 'sampling_frame'
n_runs(x, ...)
```

## Arguments

- x:

  An object containing temporal structure (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Integer representing the total number of runs

## See also

[`get_run_lengths`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_lengths.md)
for individual run lengths,
[`n_timepoints`](https://bbuchsbaum.github.io/fmridataset/reference/n_timepoints.md)
for total timepoints

## Examples

``` r
# \donttest{
# Create a sampling frame with 3 runs
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
n_runs(sf) # Returns: 3
#> [1] 3
# }
```
