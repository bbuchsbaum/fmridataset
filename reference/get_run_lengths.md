# Get Run Lengths from Sampling Frame

Extracts the lengths of individual runs/blocks from objects containing
temporal structure information.

## Usage

``` r
get_run_lengths(x, ...)

# S3 method for class 'matrix_dataset'
get_run_lengths(x, ...)

# S3 method for class 'fmri_dataset'
get_run_lengths(x, ...)

# S3 method for class 'fmri_mem_dataset'
get_run_lengths(x, ...)

# S3 method for class 'fmri_file_dataset'
get_run_lengths(x, ...)

# S3 method for class 'fmri_study_dataset'
get_run_lengths(x, ...)

# S3 method for class 'sampling_frame'
get_run_lengths(x, ...)
```

## Arguments

- x:

  An object containing temporal structure (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Integer vector where each element represents the number of timepoints in
the corresponding run

## Details

This function is synonymous with
[`blocklens`](https://bbuchsbaum.github.io/fmridataset/reference/blocklens.md)
but uses terminology more common in fMRI analysis. Each run represents a
continuous acquisition period, and the run length is the number of
timepoints (volumes) in that run.

## See also

[`blocklens`](https://bbuchsbaum.github.io/fmridataset/reference/blocklens.md)
for equivalent function,
[`n_runs`](https://bbuchsbaum.github.io/fmridataset/reference/n_runs.md)
for number of runs,
[`n_timepoints`](https://bbuchsbaum.github.io/fmridataset/reference/n_timepoints.md)
for total timepoints

## Examples

``` r
# \donttest{
# Create a sampling frame with 3 runs
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
get_run_lengths(sf) # Returns: c(100, 120, 110)
#> [1] 100 120 110
# }
```
