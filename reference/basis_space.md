# Construct a linear basis feature space

A `basis_space` is a representational feature axis linked to one parent
spatial space. Its encoder maps parent features to component
coefficients; its optional decoder maps coefficients back to the parent.
Fitting and model-specific offsets remain the responsibility of packages
such as `fmrilatent`.

## Usage

``` r
basis_space(
  parent,
  component_ids,
  encoder,
  decoder = NULL,
  data = NULL,
  basis_type = "linear_basis",
  provenance = list(),
  tolerance = 1e-08,
  metadata = list()
)
```

## Arguments

- parent:

  Parent `feature_space` represented by the basis.

- component_ids:

  Stable component identifiers.

- encoder:

  Component-by-parent analysis operator.

- decoder:

  Optional parent-by-component synthesis operator.

- data:

  One metadata row per component.

- basis_type:

  Stable basis-family label.

- provenance:

  Serializable derivation metadata.

- tolerance:

  Maximum absolute error permitted when validating that
  `encoder %*% decoder` is the component-space identity.

- metadata:

  Additional serializable metadata.

## Value

A `basis_space`.
