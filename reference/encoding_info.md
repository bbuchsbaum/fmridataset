# Get Encoding Metadata from a Latent-Mode BIDS H5 Dataset

Returns encoding metadata (family, parameters, number of components)
stored in the `/latent_meta/` group of a latent-mode BIDS HDF5 archive.
Returns `NULL` for parcellated-mode archives.

## Usage

``` r
encoding_info(x, ...)

# S3 method for class 'bids_h5_study_dataset'
encoding_info(x, ...)
```

## Arguments

- x:

  A `bids_h5_study_dataset` object.

- ...:

  Additional arguments passed to methods.

## Value

A named list with elements `encoding_family`, `encoding_params`, and
`n_components`, or `NULL`.
