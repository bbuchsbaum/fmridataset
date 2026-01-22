# Constructor for fmri_series objects

Constructor for fmri_series objects

## Usage

``` r
new_fmri_series(data, voxel_info, temporal_info, selection_info, dataset_info)
```

## Arguments

- data:

  A lazy matrix (e.g., `delarr`), `DelayedMatrix`, or base matrix

- voxel_info:

  A data.frame containing spatial metadata for each voxel

- temporal_info:

  A data.frame containing metadata for each timepoint

- selection_info:

  A list describing how the data were selected

- dataset_info:

  A list describing the source dataset and backend

## Value

An object of class `fmri_series`
