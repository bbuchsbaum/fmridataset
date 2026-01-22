# Reduce over subjects in a single pass

Reduce over subjects in a single pass

## Usage

``` r
group_reduce(
  gd,
  .map,
  .reduce,
  .init,
  order_by = NULL,
  on_error = c("stop", "warn", "skip"),
  ...
)
```

## Arguments

- gd:

  An `fmri_group`.

- .map:

  Function applied to each subject row. Should return an object that can
  be combined by `.reduce`.

- .reduce:

  Binary function combining the accumulator and the mapped value.

- .init:

  Initial accumulator value.

- order_by:

  Optional ordering column.

- on_error:

  Error handling policy: "stop", "warn", or "skip".

- ...:

  Additional arguments passed to `.map`.

## Value

The reduced value after visiting all subjects.
