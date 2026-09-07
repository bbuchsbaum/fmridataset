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

## Examples

``` r
surface_space(
  vertex_ids = c("L-1", "L-2", "L-3"),
  hemisphere = rep("left", 3)
)
#> $vertex_ids
#> [1] "L-1" "L-2" "L-3"
#> 
#> $hemisphere
#> [1] "left" "left" "left"
#> 
#> $support
#> [1] 1 2 3
#> 
#> $medial_wall
#> [1] FALSE FALSE FALSE
#> 
#> $topology
#> $topology$reference
#> NULL
#> 
#> $topology$digest
#> [1] "c251941577ab73e34bf946561f9c11708ce03c301ec36bac5e74366185f3f5f4"
#> 
#> $topology$data
#> NULL
#> 
#> 
#> $geometry
#> $geometry$reference
#> NULL
#> 
#> $geometry$digest
#> [1] "c251941577ab73e34bf946561f9c11708ce03c301ec36bac5e74366185f3f5f4"
#> 
#> $geometry$data
#> NULL
#> 
#> 
#> $template
#> NULL
#> 
#> $units
#> [1] "mm"
#> 
#> $surf_to_world
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    0
#> [2,]    0    1    0    0
#> [3,]    0    0    1    0
#> [4,]    0    0    0    1
#> 
#> $metadata
#> list()
#> 
#> $schema_version
#> [1] 2
#> 
#> attr(,"class")
#> [1] "surface_space" "feature_space"
```
