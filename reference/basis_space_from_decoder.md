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
  metadata = list()
)
```

## Arguments

- parent, component_ids, decoder, data, basis_type, provenance,
  tolerance, metadata:

  Passed to
  [`basis_space()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space.md).

## Value

A `basis_space`.
