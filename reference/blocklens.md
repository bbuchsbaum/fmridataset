# Get Block Lengths from Objects

Generic function to extract block/run lengths from various objects.
Extends the sampling_frame generic to work with dataset objects.

## Usage

``` r
blocklens(x, ...)

# S3 method for class 'matrix_dataset'
blocklens(x, ...)

# S3 method for class 'fmri_dataset'
blocklens(x, ...)

# S3 method for class 'fmri_mem_dataset'
blocklens(x, ...)

# S3 method for class 'fmri_file_dataset'
blocklens(x, ...)

# S3 method for class 'fmri_study_dataset'
blocklens(x, ...)

# S3 method for class 'sampling_frame'
blocklens(x, ...)
```

## Arguments

- x:

  An object with block structure (e.g., sampling_frame, fmri_dataset)

- ...:

  Additional arguments passed to methods

## Value

Integer vector where each element represents the number of timepoints in
the corresponding run/block

## Details

In fMRI experiments, data is often collected in multiple runs or blocks.
This function extracts the length (number of timepoints) of each run.
The sum of block lengths equals the total number of timepoints.

## See also

[`n_runs`](https://bbuchsbaum.github.io/fmridataset/reference/n_runs.md)
for number of runs,
[`n_timepoints`](https://bbuchsbaum.github.io/fmridataset/reference/n_timepoints.md)
for total timepoints,
[`get_run_lengths`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_lengths.md)
for alternative function name

## Examples

``` r
# \donttest{
# Create a dataset with 3 runs
sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
blocklens(sf) # c(100, 120, 110)
#> [1] 100 120 110

# Create dataset with multiple runs
mat <- matrix(rnorm(330 * 50), nrow = 330, ncol = 50)
ds <- matrix_dataset(mat, TR = 2, run_length = c(100, 120, 110))
blocklens(ds) # c(100, 120, 110)
#> [1] 100 120 110
# }
```
