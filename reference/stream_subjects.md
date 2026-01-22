# Stream subjects with optional ordering

Stream subjects with optional ordering

## Usage

``` r
stream_subjects(gd, prefetch = 1L, order_by = NULL)
```

## Arguments

- gd:

  An `fmri_group`.

- prefetch:

  Number of subjects to prefetch. Currently only `1L` is supported;
  higher values are accepted for future compatibility but do not change
  behaviour.

- order_by:

  Optional column name used to order subjects.

## Value

An iterator identical to
[`iter_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/iter_subjects.md).
