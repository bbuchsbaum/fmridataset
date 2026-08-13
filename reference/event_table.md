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
