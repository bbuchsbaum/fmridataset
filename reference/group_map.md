# Map a function over subjects in an fmri_group

Map a function over subjects in an fmri_group

## Usage

``` r
group_map(
  gd,
  .f,
  ...,
  out = c("list", "bind_rows"),
  order_by = NULL,
  on_error = c("stop", "warn", "skip")
)
```

## Arguments

- gd:

  An `fmri_group`.

- .f:

  A function with signature `function(row, ...)` where `row` is a
  one-row `data.frame` corresponding to a single subject.

- ...:

  Additional arguments passed through to `.f`.

- out:

  Either "list" (default) or "bind_rows" describing how to collect
  outputs.

- order_by:

  Optional column name used to define iteration order.

- on_error:

  One of "stop", "warn", or "skip" describing how to handle errors
  raised by `.f`.

## Value

Either a list (for `out = "list"`) or a bound table (for
`out = "bind_rows"`).
