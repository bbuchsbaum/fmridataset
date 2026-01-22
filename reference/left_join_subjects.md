# Left join additional subject metadata

Left join additional subject metadata

## Usage

``` r
left_join_subjects(gd, y, by = NULL, ...)
```

## Arguments

- gd:

  An `fmri_group`.

- y:

  A data frame containing additional subject-level columns.

- by:

  Character vector of join keys (defaults to the group id column).

- ...:

  Additional arguments passed to
  [`dplyr::left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
  when available.

## Value

An `fmri_group` with metadata from `y` attached.
