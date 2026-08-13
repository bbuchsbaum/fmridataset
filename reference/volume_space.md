# Construct a packed volumetric feature space

Construct a packed volumetric feature space

## Usage

``` r
volume_space(
  dim,
  affine = diag(4),
  support = NULL,
  template = NULL,
  units = "mm",
  metadata = list()
)
```

## Arguments

- dim:

  Three spatial dimensions.

- affine:

  A 4 by 4 voxel-to-world affine.

- support:

  Logical full-volume support or packed linear indices.

- template:

  Optional template/native-space identity.

- units:

  Spatial units.

- metadata:

  Additional serializable metadata.

## Value

A `volume_space`.
