# Developer tool: inject deterministic source failures

This source is exported solely for deterministic backend, codec,
cleanup, and recovery conformance tests.

## Usage

``` r
fault_source(
  source,
  stage = c("read", "open", "native_read", "close"),
  message = NULL
)
```

## Arguments

- source:

  An array source.

- stage:

  One of `"open"`, `"read"`, `"native_read"`, or `"close"`.

- message:

  Failure message.

## Value

A serializable fault-injecting source.

## Details

This is developer-only test instrumentation. Never persist a
`fault_source` as study data or use one in an analysis plan.

## Examples

``` r
src <- fault_source(memory_source(matrix(seq_len(6), nrow = 2)), stage = "read")
tryCatch(source_read(src), error = function(e) conditionMessage(e))
#> [1] "Injected read failure"
```
