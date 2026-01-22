# Create a NIfTI Backend

Create a NIfTI Backend

## Usage

``` r
nifti_backend(
  source,
  mask_source,
  preload = FALSE,
  mode = c("normal", "bigvec", "mmap", "filebacked"),
  dummy_mode = FALSE
)
```

## Arguments

- source:

  Character vector of file paths or list of in-memory NeuroVec objects

- mask_source:

  File path to mask or in-memory NeuroVol object

- preload:

  Logical, whether to eagerly load data into memory

- mode:

  Storage mode for file-backed data: 'normal', 'bigvec', 'mmap', or
  'filebacked'

- dummy_mode:

  Logical, if TRUE allows non-existent files (for testing). Default
  FALSE.

## Value

A nifti_backend S3 object
