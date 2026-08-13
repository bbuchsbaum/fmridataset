# Access derived hierarchy index data

Access derived hierarchy index data

## Usage

``` r
hierarchy_ids(x)

hierarchy_groups(x)

hierarchy_levels(x)

hierarchy_relations(x)

hierarchy_complete(x)
```

## Arguments

- x:

  An `fmri_hierarchy_index`.

## Value

`hierarchy_ids()` returns stable entity IDs; `hierarchy_groups()`
returns stable integer grouping codes; `hierarchy_levels()` and
`hierarchy_relations()` return named character vectors;
`hierarchy_complete()` returns a logical vector.
