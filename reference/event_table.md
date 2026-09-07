# Construct a keyed event table

Event rows retain their natural cardinality and are not expanded to
acquired volumes. Entity-key columns are validated against a study when
attached.

## Usage

``` r
event_table(data, key = "event_id", metadata = list())
```

## Arguments

- data:

  Scalar event annotations.

- key:

  Stable event-key column.

- metadata:

  Serializable event-table metadata.

## Value

An `fmri_event_table`.

## Examples

``` r
et <- event_table(
  data.frame(event_id = c("e1", "e2"), onset = c(0, 10), duration = c(2, 2)),
  key = "event_id"
)
event_data(et)
#> # A tibble: 2 × 3
#>   event_id onset duration
#>   <chr>    <dbl>    <dbl>
#> 1 e1           0        2
#> 2 e2          10        2
event_key(et)
#> [1] "event_id"
```
