# Apply one validity relation lazily to frame assays

Apply one validity relation lazily to frame assays

## Usage

``` r
apply_feature_validity(x, name = NULL, assays = NULL)
```

## Arguments

- x:

  An `fmri_frame` or view.

- name:

  Validity relation name.

- assays:

  Assay names to mask. Defaults to all assays.

## Value

A new frame with lazy `NA` masking and derivation provenance.
