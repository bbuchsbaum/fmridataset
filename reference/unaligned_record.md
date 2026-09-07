# Construct a typed unaligned metadata record

Container metadata is deliberately unaligned. Array- or table-valued
data belong in assays, axis blocks, relations, typed tables, or linked
frames. When `domains` are supplied, vector lengths matching an
observation, feature, or entity domain are rejected as probable hidden
alignment.

## Usage

``` r
unaligned_record(x = list(), domains = NULL)
```

## Arguments

- x:

  A named serializable list.

- domains:

  Optional named observation, feature, or entity sizes used to detect
  hidden alignment.

## Value

An `unaligned_record`.

## Examples

``` r
rec <- unaligned_record(list(task = "rest", tr = 2))
rec$task
#> [1] "rest"
```
