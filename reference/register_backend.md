# Register a Storage Backend

Registers a new storage backend type in the global registry. External
packages can use this function to add custom backend support.

## Usage

``` r
register_backend(
  name,
  factory,
  description = NULL,
  validate_function = NULL,
  overwrite = FALSE
)
```

## Arguments

- name:

  Character string, unique name for the backend type

- factory:

  Function that creates backend instances, must accept `...` arguments

- description:

  Optional character string describing the backend

- validate_function:

  Optional function to validate backend instances beyond the standard
  contract

- overwrite:

  Logical, whether to overwrite existing registration (default: FALSE)

## Value

Invisibly returns TRUE on successful registration

## Details

The factory function should:

- Accept all necessary parameters to create a backend instance

- Return an object that inherits from "storage_backend"

- Implement all required storage backend methods

The validate_function should:

- Accept a backend object as first argument

- Return TRUE if valid, or throw informative error if invalid

- Perform any backend-specific validation beyond the standard contract

## Examples

``` r
if (FALSE) { # \dontrun{
# Register a custom backend
my_backend_factory <- function(source, ...) {
  # Create and return backend instance
  backend <- list(source = source, ...)
  class(backend) <- c("my_backend", "storage_backend")
  backend
}

register_backend(
  name = "my_backend",
  factory = my_backend_factory,
  description = "Custom backend for my data format"
)
} # }
```
