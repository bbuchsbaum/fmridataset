# Construct a typed auxiliary table

Construct a typed auxiliary table

## Usage

``` r
auxiliary_table(data, key = NULL, role = "auxiliary", metadata = list())
```

## Arguments

- data:

  Scalar tabular data.

- key:

  Optional stable unique-key column.

- role:

  Stable table role such as `"files"`, `"contrasts"`, or `"transforms"`.

- metadata:

  Unaligned table-level metadata.

## Value

An `fmri_auxiliary_table`.

## Examples

``` r
at <- auxiliary_table(
  data.frame(contrast = c("A-B", "B-A"), stat = c(2.1, -2.1)),
  key = "contrast", role = "contrasts"
)
table_data(at)
#> # A tibble: 2 × 2
#>   contrast  stat
#>   <chr>    <dbl>
#> 1 A-B        2.1
#> 2 B-A       -2.1
```
