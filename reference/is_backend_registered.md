# Check if Backend is Registered

Tests whether a backend type is registered in the system.

## Usage

``` r
is_backend_registered(name)
```

## Arguments

- name:

  Character string, name of backend to check

## Value

Logical, TRUE if backend is registered

## Examples

``` r
is_backend_registered("nifti") # TRUE (built-in)
#> [1] TRUE
is_backend_registered("custom") # FALSE (unless registered)
#> [1] FALSE
```
