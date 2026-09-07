# Construct a basis space from a synthesis dictionary

Computes the exact unregularized least-squares encoder `(D' D)^{-1} D'`
for a full-column-rank decoder `D`.

## Usage

``` r
basis_space_from_decoder(
  parent,
  component_ids,
  decoder,
  data = NULL,
  basis_type = "linear_basis",
  provenance = list(),
  tolerance = 1e-08,
  metadata = list(),
  encoder = c("least_squares", "none")
)
```

## Arguments

- parent, component_ids, decoder, data, basis_type, provenance,
  tolerance, metadata:

  Passed to
  [`basis_space()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space.md).

- encoder:

  Either `"least_squares"` to construct and validate the exact
  unregularized left inverse, or `"none"` for a synthesis-only basis.

## Value

A `basis_space`.

## Examples

``` r
parent <- volume_space(c(2, 1, 1), support = 1:2)
x <- basis_space_from_decoder(parent, c("c1", "c2"), diag(2))
basis_synthesis(x)
#>      [,1] [,2]
#> [1,]    1    0
#> [2,]    0    1
```
