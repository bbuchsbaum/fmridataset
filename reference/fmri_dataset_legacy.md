# Legacy fMRI Dataset Constructor

Backward compatibility wrapper for fmri_dataset. This function provides
the same interface as the original fmri_dataset function.

## Usage

``` r
fmri_dataset_legacy(scans, mask, TR, run_length, preload = FALSE, ...)
```

## Arguments

- scans:

  Either a character vector of file paths to scans or a list of NeuroVec
  objects

- mask:

  Either a character file path to a mask or a NeuroVol mask object

- TR:

  The repetition time

- run_length:

  Numeric vector of run lengths

- preload:

  Whether to preload data into memory

- ...:

  Additional arguments passed to fmri_dataset

## Value

An fmri_dataset object

## Examples

``` r
if (FALSE) { # \dontrun{
# Create dataset from files
dset <- fmri_dataset_legacy(
  scans = c("scan1.nii", "scan2.nii"),
  mask = "mask.nii",
  TR = 2,
  run_length = c(100, 100)
)
} # }
```
