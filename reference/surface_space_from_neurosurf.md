# Adapt a neurosurf geometry to a surface feature space

`neurosurf` remains the owner of mesh geometry and algorithms. This
adapter extracts its stable topology, coordinates, hemisphere, and world
transform into the backend-neutral identity required by an `fmri_frame`.

## Usage

``` r
surface_space_from_neurosurf(
  geometry,
  vertex_ids = NULL,
  support = NULL,
  medial_wall = NULL,
  template = NULL,
  units = "mm",
  metadata = list()
)
```

## Arguments

- geometry:

  A
  [`neurosurf::SurfaceGeometry`](https://bbuchsbaum.github.io/neurosurf/reference/SurfaceGeometry.html).

- vertex_ids:

  Optional stable full-mesh vertex IDs.

- support, medial_wall, template, units, metadata:

  Passed to
  [`surface_space()`](https://bbuchsbaum.github.io/fmridataset/reference/surface_space.md).

## Value

A `surface_space`.
