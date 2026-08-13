# Inject deterministic source failures

Inject deterministic source failures

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
