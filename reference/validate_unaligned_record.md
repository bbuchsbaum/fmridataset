# Validate a typed unaligned metadata record

Validate a typed unaligned metadata record

## Usage

``` r
validate_unaligned_record(x, domains = NULL)
```

## Arguments

- x:

  An `unaligned_record`.

- domains:

  Optional named alignment-domain sizes.

## Value

`x`, invisibly.

## Examples

``` r
rec <- unaligned_record(list(task = "rest", tr = 2))
validate_unaligned_record(rec)
```
