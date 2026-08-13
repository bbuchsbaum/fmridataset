# Construct a packed cortical surface feature space

Construct a packed cortical surface feature space

## Usage

``` r
surface_space(
  vertex_ids,
  hemisphere,
  support = NULL,
  topology = NULL,
  geometry = NULL,
  medial_wall = NULL,
  template = NULL,
  units = "mm",
  surf_to_world = diag(4),
  metadata = list()
)
```

## Arguments

- vertex_ids:

  Stable IDs for every vertex in the full mesh.

- hemisphere:

  One `"left"` or `"right"` label per full-mesh vertex.

- support:

  Active vertex positions or IDs. By default, all non-medial-wall
  vertices are active.

- topology:

  A three-column face matrix or asset descriptor with `reference`,
  `digest`, and optional `data`/`faces`.

- geometry:

  A vertex-by-three coordinate matrix or asset descriptor with
  `reference`, `digest`, and optional `data`/`coordinates`.

- medial_wall:

  Logical full-mesh medial-wall mask.

- template:

  Optional template identity such as `"fsLR-32k"`.

- units:

  Coordinate units.

- surf_to_world:

  A finite 4 by 4 surface-to-world transform, following the
  [`neurosurf::SurfaceGeometry`](https://bbuchsbaum.github.io/neurosurf/reference/SurfaceGeometry.html)
  convention.

- metadata:

  Additional serializable metadata.

## Value

A `surface_space`.
