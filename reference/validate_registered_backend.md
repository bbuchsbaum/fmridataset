# Validate Registered Backend

Validates a backend instance using both the standard contract and any
backend-specific validation function.

## Usage

``` r
validate_registered_backend(backend, registration = NULL)
```

## Arguments

- backend:

  A storage backend object

- registration:

  Backend registration information (optional, will be looked up if not
  provided)

## Value

TRUE if valid, otherwise throws an error
