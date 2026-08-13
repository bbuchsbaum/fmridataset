# Adapt fmrilatent spatial loadings to a basis feature space

`fmrilatent` remains the owner of latent fitting, temporal scores,
handles, and offsets. This adapter extracts only its spatial synthesis
dictionary and constructs the corresponding least-squares feature-space
algebra.

## Usage

``` r
basis_space_from_fmrilatent(
  x,
  parent,
  component_ids = NULL,
  data = NULL,
  provenance = list(),
  tolerance = 1e-08
)
```

## Arguments

- x:

  An explicit `fmrilatent` object with
  [`loadings()`](https://rdrr.io/r/stats/loadings.html).

- parent:

  Parent feature space aligned to the loading rows.

- component_ids:

  Optional stable component IDs.

- data:

  Optional component metadata.

- provenance:

  Additional serializable provenance.

- tolerance:

  Left-inverse validation tolerance.

## Value

A `basis_space`.
