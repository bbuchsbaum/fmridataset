# Stop with a Custom Error

Stop with a Custom Error

## Usage

``` r
stop_fmridataset(error_fn, message = NULL, ...)
```

## Arguments

- error_fn:

  Error constructor function

- message:

  Error message (optional if provided as first ... argument)

- ...:

  Arguments passed to the error constructor

## Value

Does not return; always signals the constructed condition with
[`stop()`](https://rdrr.io/r/base/stop.html).

## Examples

``` r
tryCatch(
  fmridataset:::stop_fmridataset(fmridataset:::fmridataset_error_config, "bad value"),
  fmridataset_error_config = function(e) conditionMessage(e)
)
#> [1] "bad value"
```
