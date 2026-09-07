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

## Examples

``` r
r1 <- provenance_record("load", inputs = list(path = "toy.nii"))
g <- provenance_graph(r1)
r2 <- provenance_record("normalize", parents = r1$id)
g <- append_provenance(g, r2)
provenance_tips(g)
#> [1] "8580b26ec65e16250a3c3174995f6fbe70b9ba6d8ede6b8638eb5630bfa4c13f"
```
