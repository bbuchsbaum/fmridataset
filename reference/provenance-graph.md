# Construct and inspect an immutable provenance graph

Construct and inspect an immutable provenance graph

## Usage

``` r
provenance_graph(...)

validate_provenance_graph(x)

provenance_records(x)

provenance_tips(x)

provenance_digest(x)

append_provenance(x, records)
```

## Arguments

- ...:

  `provenance_record` objects, or one list of records.

- x:

  A `provenance_graph`.

- records:

  One or more records appended to `x`.

## Value

A validated `provenance_graph`, its records, tips, or digest.
