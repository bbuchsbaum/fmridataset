# Developer tool: instrument an array source

`counting_source()` records numerical reads without placing a mutable
environment inside the source descriptor. It is exported for backend and
downstream conformance suites, not as an application data source.

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

## Details

This is developer-only test instrumentation. The counter registry is
process-local, is not persisted with the descriptor, and must not be
used as provenance or as an execution receipt.

## Examples

``` r
src <- counting_source(memory_source(matrix(seq_len(6), nrow = 2)))
source_read(src, observations = 1)
#>      [,1] [,2] [,3]
#> [1,]    1    3    5
source_counts(src)$reads
#> [1] 1
```
