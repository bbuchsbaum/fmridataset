# Get Parcellation Information from a BIDS H5 Dataset

Generic function to retrieve parcellation metadata from a BIDS HDF5
dataset stored in parcellated mode.

## Usage

``` r
parcellation_info(x, ...)

# S3 method for class 'bids_h5_study_dataset'
parcellation_info(x, ...)
```

## Arguments

- x:

  A BIDS H5 study dataset object

- ...:

  Additional arguments passed to methods

## Value

A list with elements: cluster_ids, cluster_map, labels, n_parcels
