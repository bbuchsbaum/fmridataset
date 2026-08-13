# Summarize feature coverage without imposing an analysis policy

Summarize feature coverage without imposing an analysis policy

## Usage

``` r
validity_coverage(x, name = NULL, domain = c("entity", "observation"))
```

## Arguments

- x:

  An `fmri_frame` or view.

- name:

  Validity relation name.

- domain:

  Weight unique entities or frame observations.

## Value

Named fraction-valid vector over frame features.
