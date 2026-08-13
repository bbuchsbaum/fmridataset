# Filter every study representation through one shared entity selection

Filter every study representation through one shared entity selection

## Usage

``` r
filter_entities(x, entity, predicate)
```

## Arguments

- x:

  An `fmri_study` or filtered view.

- entity:

  Bare or quoted shared entity name.

- predicate:

  A scalar-metadata predicate evaluated on that entity table.

## Value

A lazy `fmri_study_view`.
