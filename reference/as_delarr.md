# Convert backend to a delarr lazy matrix

Provides a lightweight S3 interface that defers materialization of
backend data. The returned object is compatible with
[`delarr::collect()`](https://rdrr.io/pkg/delarr/man/collect.html) as
well as base [`as.matrix()`](https://rdrr.io/r/base/matrix.html) for
realization.

## Usage

``` r
as_delarr(backend, ...)

# S3 method for class 'matrix_backend'
as_delarr(backend, ...)

# S3 method for class 'nifti_backend'
as_delarr(backend, ...)

# S3 method for class 'study_backend'
as_delarr(backend, ...)

# Default S3 method
as_delarr(backend, ...)
```

## Arguments

- backend:

  A storage backend object

- ...:

  Passed to methods

## Value

A `delarr` lazy matrix
