# Get TR (Repetition Time) from Sampling Frame

Extracts the repetition time (TR) in seconds from objects containing
temporal information about fMRI acquisitions.

## Usage

``` r
get_TR(x, ...)

# S3 method for class 'matrix_dataset'
get_TR(x, ...)

# S3 method for class 'fmri_dataset'
get_TR(x, ...)

# S3 method for class 'fmri_mem_dataset'
get_TR(x, ...)

# S3 method for class 'fmri_file_dataset'
get_TR(x, ...)

# S3 method for class 'fmri_study_dataset'
get_TR(x, ...)

# S3 method for class 'sampling_frame'
get_TR(x, ...)
```

## Arguments

- x:

  An object containing temporal information (e.g., sampling_frame,
  fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Numeric value representing TR in seconds

## Details

The TR (repetition time) is the time between successive acquisitions of
the same slice in an fMRI scan, typically measured in seconds. This
parameter is crucial for temporal analyses and hemodynamic modeling.

## See also

`sampling_frame` for creating temporal structures,
[`get_total_duration`](https://bbuchsbaum.github.io/fmridataset/reference/get_total_duration.md)
for total scan duration

## Examples

``` r
# \donttest{
# Create a sampling frame with TR = 2 seconds
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
get_TR(sf) # Returns: 2
#> [1] 2
# }
```
