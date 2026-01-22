# Create an H5 Backend

Create an H5 Backend

## Usage

``` r
h5_backend(
  source,
  mask_source,
  mask_dataset = "data/elements",
  data_dataset = "data",
  preload = FALSE
)
```

## Arguments

- source:

  Character vector of file paths to H5 files or list of H5NeuroVec
  objects

- mask_source:

  File path to H5 mask file, H5 file containing mask, or in-memory
  NeuroVol object

- mask_dataset:

  Character string specifying the dataset path within H5 file for mask
  (default: "data/elements")

- data_dataset:

  Character string specifying the dataset path within H5 files for data
  (default: "data")

- preload:

  Logical, whether to eagerly load H5NeuroVec objects into memory

## Value

An h5_backend S3 object
