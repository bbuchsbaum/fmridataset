# Configuration Error

Raised when invalid configuration is provided to a backend or dataset.

## Usage

``` r
fmridataset_error_config(message, parameter = NULL, value = NULL, ...)
```

## Arguments

- message:

  Character string describing the configuration error

- parameter:

  The parameter that was invalid

- value:

  The invalid value provided

- ...:

  Additional context

## Value

A configuration error condition

## Examples

``` r
cond <- fmridataset:::fmridataset_error_config("bad value", parameter = "chunks")
cond$parameter
#> [1] "chunks"
```
