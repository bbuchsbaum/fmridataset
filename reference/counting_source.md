# Instrument an array source

`counting_source()` records numerical reads without placing a mutable
environment inside the source descriptor.

## Usage

``` r
counting_source(source)

source_counts(x)

reset_source_counts(x)
```

## Arguments

- source:

  An array source.

- x:

  A counting source.

## Value

A serializable instrumented source.
