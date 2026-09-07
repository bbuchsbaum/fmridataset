# Null-coalescing operator

If x is NULL, return y; otherwise return x

## Usage

``` r
x %||% y
```

## Arguments

- x:

  A value to test for `NULL`.

- y:

  The fallback value returned when `x` is `NULL`.

## Value

`y` if `x` is `NULL`; otherwise `x`.

## Examples

``` r
fmridataset:::`%||%`(NULL, 1)
#> [1] 1
fmridataset:::`%||%`(2, 1)
#> [1] 2
```
