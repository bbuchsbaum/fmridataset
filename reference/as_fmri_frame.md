# Coerce an object to the canonical frame protocol

Companion packages implement methods for legacy or domain-specific
objects. The generic is owned by `fmridataset` so conversions have one
canonical destination and do not introduce competing frame classes.

## Usage

``` r
as_fmri_frame(x, ...)
```

## Arguments

- x:

  An object convertible to an `fmri_frame`.

- ...:

  Method-specific arguments.

## Value

An `fmri_frame`.
