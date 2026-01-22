# Unregister a Backend

Removes a backend from the registry. Use with caution as this may break
code that depends on the backend.

## Usage

``` r
unregister_backend(name)
```

## Arguments

- name:

  Character string, name of backend to remove

## Value

Invisibly returns TRUE if backend was removed, FALSE if it wasn't
registered

## Examples

``` r
if (FALSE) { # \dontrun{
# Remove a custom backend
unregister_backend("my_custom_backend")
} # }
```
