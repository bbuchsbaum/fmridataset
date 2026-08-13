# Inspect and validate an array-source contract

A valid canonical source is two-dimensional, has an explicit supported
dtype and chunk grid, advertises serializable block slicing, provides a
stable non-empty fingerprint, and contains no runtime handles or
closures.

## Usage

``` r
source_descriptor(x)

validate_array_source(x)
```

## Arguments

- x:

  An `array_source` descriptor.

## Value

`source_descriptor()` returns a plain serializable contract list.
`validate_array_source()` invisibly returns `x` or raises a structured
source-contract error.
