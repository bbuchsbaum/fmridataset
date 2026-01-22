# Backend I/O Error

Raised when a storage backend encounters read/write failures.

## Usage

``` r
fmridataset_error_backend_io(message, file = NULL, operation = NULL, ...)
```

## Arguments

- message:

  Character string describing the I/O error

- file:

  Path to the file that caused the error (optional)

- operation:

  The operation that failed (e.g., "read", "write")

- ...:

  Additional context

## Value

A backend I/O error condition
