# Build the structured conditions an array source may signal

Storage packages implement the [array
source](https://bbuchsbaum.github.io/fmridataset/reference/array-source.md)
protocol and must fail the way built-in sources fail, so that callers
can handle every source alike. `source_error()` builds the condition
object for the three failure kinds the protocol defines; signal it with
[`stop()`](https://rdrr.io/r/base/stop.html).

## Usage

``` r
source_error(message, type = c("stale", "io", "contract"), ...)
```

## Arguments

- message:

  One-sentence description of the failure.

- type:

  One of `"stale"`, `"io"`, or `"contract"`.

- ...:

  Named fields stored on the condition, such as `source`, `expected`,
  `actual`, `file`, `operation`, or `field`.

## Value

A condition object inheriting from the type-specific class,
`fmridataset_error`, `error`, and `condition`.

## Details

- `"stale"` (`fmridataset_error_source_stale`): the physical store no
  longer matches the descriptor's revision evidence (a file was
  rewritten, a store was replaced). Supply `source`, `expected`, and
  `actual` so the caller can report what changed.

- `"io"` (`fmridataset_error_backend_io`): a genuine read, open, or
  close failure. Supply `file` and `operation` where known.

- `"contract"` (`fmridataset_error_source_contract`): the descriptor or
  a method violates the protocol. Supply `field`.

Every condition also carries the `fmridataset_error` class, so a single
handler can catch all package errors.

## Examples

``` r
cond <- source_error(
  "Store was rewritten after the descriptor was built.",
  type = "stale", source = "example.h5",
  expected = "abc", actual = "def"
)
inherits(cond, "fmridataset_error_source_stale")
#> [1] TRUE
tryCatch(stop(cond), fmridataset_error_source_stale = function(e) e$actual)
#> [1] "def"
```
