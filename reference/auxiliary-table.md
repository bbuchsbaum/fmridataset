# Validate and inspect a typed auxiliary table

Validate and inspect a typed auxiliary table

## Usage

``` r
validate_auxiliary_table(x)

table_data(x)

table_key(x)

table_role(x)
```

## Arguments

- x:

  An `fmri_auxiliary_table`.

## Value

`validate_auxiliary_table()` returns `x` invisibly; other functions
return table data, key, or role.

## Examples

``` r
at <- auxiliary_table(
  data.frame(contrast = c("A-B", "B-A"), stat = c(2.1, -2.1)),
  key = "contrast", role = "contrasts"
)
validate_auxiliary_table(at)
table_key(at)
#> [1] "contrast"
table_role(at)
#> [1] "contrasts"
```
