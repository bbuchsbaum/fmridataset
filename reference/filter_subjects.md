# Filter subjects in an fmri_group

Expressions are evaluated in the context of `subjects(gd)` and may refer
to its columns directly. Multiple expressions are combined with logical
AND.

## Usage

``` r
filter_subjects(gd, ...)
```

## Arguments

- gd:

  An `fmri_group`.

- ...:

  Logical expressions used to filter rows.

## Value

An updated `fmri_group` containing only the rows that satisfy all
predicates.
