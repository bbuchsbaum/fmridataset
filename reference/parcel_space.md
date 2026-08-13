# Construct a parent-linked parcel feature space

A `parcel_space` owns only the feature-axis algebra required by
`fmridataset`: stable parcel identity, a parent feature space, and
explicit parent-to-parcel operators. Atlas discovery, labels, and
provenance remain owned by `neuroatlas` and can be imported with
[`parcel_space_from_atlas()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space_from_atlas.md).

## Usage

``` r
parcel_space(
  parent,
  parcel_ids,
  membership,
  data = NULL,
  atlas,
  aggregation = c("mean", "sum"),
  decoder = NULL,
  metadata = list()
)
```

## Arguments

- parent:

  Parent `feature_space` (usually a volume or surface space).

- parcel_ids:

  Stable atlas-native parcel identifiers.

- membership:

  Non-negative parent-feature by parcel membership weights.

- data:

  One metadata row per parcel. The `id`, `label`, and `hemi` conventions
  match
  [`neuroatlas::parcel_data`](https://bbuchsbaum.github.io/neuroatlas/reference/parcel_data.html)
  when present.

- atlas:

  Atlas identity list containing at least `id`, or a scalar ID.

- aggregation:

  Either weighted `"mean"` or weighted `"sum"`.

- decoder:

  Optional parent-feature by parcel reconstruction operator. The default
  blends overlapping parcel values by row-normalized membership.

- metadata:

  Additional serializable metadata.

## Value

A `parcel_space`.
