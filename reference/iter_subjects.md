# Iterate subjects one-by-one (streaming)

Iterate subjects one-by-one (streaming)

## Usage

``` r
iter_subjects(gd, order_by = NULL)
```

## Arguments

- gd:

  An `fmri_group`.

- order_by:

  Optional character scalar giving the column used to order iteration.
  If supplied, subjects are iterated in ascending order of this column
  (with `NA` values placed last).

## Value

A list with a single element
[`next`](https://rdrr.io/r/base/Control.html) that yields a one-row
`data.frame` for each subject when called repeatedly. The dataset column
is automatically flattened to the underlying `fmri_dataset` object.
