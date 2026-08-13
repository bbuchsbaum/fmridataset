# Dataset Methods for fmridataset

This file implements methods for dataset objects that delegate to their
internal sampling_frame objects for temporal information.

## Details

All dataset subclasses (matrix_dataset, fmri_mem_dataset,
fmri_file_dataset, fmri_study_dataset) inherit from fmri_dataset, so the
fmri_dataset methods are dispatched automatically via S3 inheritance.
Only fmri_dataset-level methods are needed.
