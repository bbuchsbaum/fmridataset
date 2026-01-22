# Convert Backend to DelayedArray

Provides a DelayedArray interface for storage backends. The returned
object lazily retrieves data via the backend when subsets of the array
are accessed.

## Usage

``` r
as_delayed_array(backend, sparse_ok = FALSE, ...)

# S3 method for class 'nifti_backend'
as_delayed_array(backend, sparse_ok = FALSE, ...)

# S3 method for class 'matrix_backend'
as_delayed_array(backend, sparse_ok = FALSE, ...)

# S3 method for class 'study_backend'
as_delayed_array(backend, sparse_ok = FALSE, ...)

# Default S3 method
as_delayed_array(backend, sparse_ok = FALSE, ...)
```

## Arguments

- backend:

  A storage backend object

- sparse_ok:

  Logical, allow sparse representation when possible

- ...:

  Additional arguments passed to methods

## Value

A DelayedArray object

## Examples

``` r
if (FALSE) { # \dontrun{
b <- matrix_backend(matrix(rnorm(20), nrow = 5))
da <- as_delayed_array(b)
dim(da)
} # }
```
