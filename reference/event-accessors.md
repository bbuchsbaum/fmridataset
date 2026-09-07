# Event-table accessors

Event-table accessors

## Usage

``` r
event_data(x)

event_key(x)
```

## Arguments

- x:

  An `fmri_event_table`.

## Value

Event scalar data or the stable key name.

## Examples

``` r
et <- event_table(
  data.frame(event_id = c("e1", "e2"), onset = c(0, 10)),
  key = "event_id"
)
event_data(et)
#> # A tibble: 2 × 2
#>   event_id onset
#>   <chr>    <dbl>
#> 1 e1           0
#> 2 e2          10
event_key(et)
#> [1] "event_id"
```
