# Create an fMRI Memory Dataset Object

This function creates an fMRI memory dataset object, which is a list
containing information about the scans, mask, TR, number of runs, event
table, base path, sampling frame, and censor.

## Usage

``` r
fmri_mem_dataset(
  scans,
  mask,
  TR,
  run_length = sapply(scans, function(x) dim(x)[4]),
  event_table = data.frame(),
  base_path = ".",
  censor = NULL
)
```

## Arguments

- scans:

  A list of objects of class `NeuroVec` from the neuroim2 package.

- mask:

  A binary mask of class `NeuroVol` from the neuroim2 package indicating
  the set of voxels to include in analyses.

- TR:

  Repetition time (TR) of the fMRI acquisition.

- run_length:

  A numeric vector specifying the length of each run in the dataset.
  Default is the length of the scans.

- event_table:

  An optional data frame containing event information. Default is an
  empty data frame.

- base_path:

  Base directory for relative file names. Absolute paths are used as-is.

- censor:

  An optional numeric vector specifying which time points to censor.
  Default is NULL.

## Value

An fMRI memory dataset object of class c("fmri_mem_dataset",
"volumetric_dataset", "fmri_dataset", "list").

## Examples

``` r
# Create a NeuroVec object
d <- c(10, 10, 10, 10)
nvec <- neuroim2::NeuroVec(array(rnorm(prod(d)), d), space = neuroim2::NeuroSpace(d))

# Create a NeuroVol mask
mask <- neuroim2::NeuroVol(array(rnorm(10 * 10 * 10), d[1:3]), space = neuroim2::NeuroSpace(d[1:3]))
mask[mask < .5] <- 0

# Create an fmri_mem_dataset
dset <- fmri_mem_dataset(list(nvec), mask, TR = 2)
```
