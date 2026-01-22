# Create an fMRI Dataset Object from H5 Files

This function creates an fMRI dataset object specifically from H5 files
using the fmristore package. Each scan is stored as an H5 file that
loads to an H5NeuroVec object.

## Usage

``` r
fmri_h5_dataset(
  h5_files,
  mask_source,
  TR,
  run_length,
  event_table = data.frame(),
  base_path = ".",
  censor = NULL,
  preload = FALSE,
  mask_dataset = "data/elements",
  data_dataset = "data"
)
```

## Arguments

- h5_files:

  A vector of one or more file paths to H5 files containing the fMRI
  data.

- mask_source:

  File path to H5 mask file, regular mask file, or in-memory NeuroVol
  object.

- TR:

  The repetition time in seconds of the scan-to-scan interval.

- run_length:

  A vector of one or more integers indicating the number of scans in
  each run.

- event_table:

  A data.frame containing the event onsets and experimental variables.
  Default is an empty data.frame.

- base_path:

  Base directory for relative file names. Absolute paths are used as-is.

- censor:

  A binary vector indicating which scans to remove. Default is NULL.

- preload:

  Read H5NeuroVec objects eagerly rather than on first access. Default
  is FALSE.

- mask_dataset:

  Character string specifying the dataset path within H5 file for mask
  (default: "data/elements").

- data_dataset:

  Character string specifying the dataset path within H5 files for data
  (default: "data").

## Value

An fMRI dataset object of class c("fmri_file_dataset",
"volumetric_dataset", "fmri_dataset", "list").

## Examples

``` r
if (FALSE) { # \dontrun{
# Create an fMRI dataset with H5NeuroVec files (standard fmristore format)
dset <- fmri_h5_dataset(
  h5_files = c("scan1.h5", "scan2.h5", "scan3.h5"),
  mask_source = "mask.h5",
  TR = 2,
  run_length = c(150, 150, 150)
)

# Create an fMRI dataset with H5 files and NIfTI mask
dset <- fmri_h5_dataset(
  h5_files = "single_scan.h5",
  mask_source = "mask.nii",
  TR = 2,
  run_length = 300
)

# Custom dataset paths (if using non-standard H5 structure)
dset <- fmri_h5_dataset(
  h5_files = "custom_scan.h5",
  mask_source = "custom_mask.h5",
  TR = 2,
  run_length = 200,
  data_dataset = "my_data_path",
  mask_dataset = "my_mask_path"
)
} # }
```
