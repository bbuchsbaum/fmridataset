# Create a Custom fmridataset Error

Create a Custom fmridataset Error

## Usage

``` r
fmridataset_error(message, class = character(), ...)
```

## Arguments

- message:

  Character string describing the error

- class:

  Character vector of error classes

- ...:

  Additional data to include in the error condition

## Value

A condition object

## Examples

``` r
cond <- fmridataset:::fmridataset_error("bad input", class = "fmridataset_error_config")
inherits(cond, "fmridataset_error")
#> [1] TRUE
```
