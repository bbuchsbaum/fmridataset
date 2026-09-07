# Coerce serializable lineage to a provenance graph

Canonical containers accept only `NULL` or a validated
`provenance_graph`. Any other serializable lineage value is wrapped,
unchanged, in a single `legacy_provenance` record so that its origin
remains inspectable rather than being silently reinterpreted.

## Usage

``` r
as_provenance_graph(x)
```

## Arguments

- x:

  `NULL`, a `provenance_graph`, or a serializable lineage value.

## Value

A validated `provenance_graph`.

## Examples

``` r
g <- as_provenance_graph(NULL)
provenance_tips(g)
#> character(0)
g2 <- as_provenance_graph(list(note = "legacy value"))
provenance_tips(g2)
#> [1] "4129efcfc9a68b8f549ffd58a8152225e4e93a61205f0fe14c781814a49b8c46"
```
