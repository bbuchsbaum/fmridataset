# Mutate subject-level attributes

Adds or modifies columns on the underlying subjects table. Expressions
are evaluated sequentially so newly created columns are available to
later expressions.

## Usage

``` r
mutate_subjects(gd, ...)
```

## Arguments

- gd:

  An `fmri_group`.

- ...:

  Logical expressions used to filter rows.

## Value

An updated `fmri_group` with modified subject attributes.
