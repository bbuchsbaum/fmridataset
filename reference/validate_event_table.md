# Validate a keyed event table

Validate a keyed event table

## Usage

``` r
validate_event_table(x)
```

## Arguments

- x:

  An `fmri_event_table`.

## Value

`x`, invisibly.

## Examples

``` r
et <- event_table(
  data.frame(event_id = c("e1", "e2"), onset = c(0, 10)),
  key = "event_id"
)
validate_event_table(et)
```
