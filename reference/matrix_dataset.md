# Matrix Dataset Constructor

This function creates a matrix dataset object, which is a list
containing information about the data matrix, TR, number of runs, event
table, sampling frame, and mask.

## Usage

``` r
matrix_dataset(datamat, TR, run_length, event_table = data.frame())
```

## Arguments

- datamat:

  A matrix where each column is a voxel time-series.

- TR:

  Repetition time (TR) of the fMRI acquisition.

- run_length:

  A numeric vector specifying the length of each run in the dataset.

- event_table:

  An optional data frame containing event information. Default is an
  empty data frame.

## Value

A matrix dataset object of class c("matrix_dataset", "fmri_dataset",
"list").

## Examples

``` r
# A matrix with 100 rows and 100 columns (voxels)
X <- matrix(rnorm(100 * 100), 100, 100)
dset <- matrix_dataset(X, TR = 2, run_length = 100)
```
