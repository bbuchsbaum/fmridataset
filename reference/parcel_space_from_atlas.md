# Build a parcel space from a neuroatlas atlas

The adapter delegates atlas identity and label interpretation to
`neuroatlas`. For surface atlases it uses `get_roi()` so atlas-specific
hemisphere-local coding is not duplicated here.

## Usage

``` r
parcel_space_from_atlas(
  atlas,
  parent,
  aggregation = c("mean", "sum"),
  metadata = list()
)
```

## Arguments

- atlas:

  A `neuroatlas` atlas or surfatlas.

- parent:

  The aligned parent `volume_space` or `surface_space`.

- aggregation:

  Aggregation method passed to
  [`parcel_space()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space.md).

- metadata:

  Serializable metadata passed to
  [`parcel_space()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space.md).

## Value

A `parcel_space` aligned to `parent`.
