# Study Backend

Composite backend that lazily combines multiple subject-level backends.

## Usage

``` r
study_backend(
  backends,
  subject_ids = NULL,
  strict = getOption("fmridataset.mask_check", "identical")
)
```

## Arguments

- backends:

  list of storage_backend objects

- subject_ids:

  vector of subject identifiers matching `backends`

- strict:

  mask validation mode. "identical" or "intersect"

## Value

A `study_backend` object
