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

## Examples

``` r
# \donttest{
# Loading fmrilatent itself takes several seconds, so this full
# integration example is wrapped in \donttest{}.
if (requireNamespace("fmrilatent", quietly = TRUE)) {
  parent <- volume_space(c(2, 2, 1), support = 1:4, template = "toy-native")
  decoder <- Matrix::Matrix(
    matrix(c(1, 0, 0, 1, 1, 1, 2, -1), nrow = 4, byrow = TRUE),
    sparse = TRUE
  )
  scores <- Matrix::Matrix(matrix(c(1, 0, 0, 1, 2, -1), nrow = 3, byrow = TRUE))
  mask <- neuroim2::LogicalNeuroVol(
    array(TRUE, dim = c(2, 2, 1)), neuroim2::NeuroSpace(c(2, 2, 1))
  )
  latent <- fmrilatent::LatentNeuroVec(
    basis = scores, loadings = decoder,
    space = neuroim2::NeuroSpace(c(2, 2, 1, 3)), mask = mask,
    offset = rep(10, 4), meta = list(family = "toy_pca")
  )
  x <- basis_space_from_fmrilatent(latent, parent = parent)
  basis_projection_info(x)
}
#> $left_inverse_validated
#> [1] TRUE
#> 
#> $left_inverse_error
#> [1] 6.661338e-16
#> 
#> $tolerance
#> [1] 1e-08
#> 
# }
```
