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

## Examples

``` r
# \donttest{
# Loading neuroatlas and neurosurf alone takes several seconds, so this
# example runs under --run-donttest rather than on every check.
if (requireNamespace("neuroatlas", quietly = TRUE) &&
  requireNamespace("neurosurf", quietly = TRUE)) {
  old_rgl <- Sys.getenv("RGL_USE_NULL", unset = NA)
  Sys.setenv(RGL_USE_NULL = "TRUE")
  left <- matrix(c(0, 0, 0, 1, 0, 0, 0, 1, 0), ncol = 3, byrow = TRUE)
  right <- sweep(left, 2, c(0, 0, 1), "+")
  faces <- matrix(c(0, 1, 2), nrow = 1)
  lh <- neurosurf::SurfaceGeometry(left, faces, "lh")
  rh <- neurosurf::SurfaceGeometry(right, faces, "rh")
  atlas <- list(
    name = "toy-surface",
    lh_atlas = neurosurf::NeuroSurface(lh, 1:3, c(1, 1, 0)),
    rh_atlas = neurosurf::NeuroSurface(rh, 1:3, c(2, 2, 0)),
    ids = 1:2, labels = c("A", "B"), orig_labels = c("A", "B"),
    hemi = c("left", "right"), network = NULL, cmap = NULL,
    surf_type = "pial", surface_space = "toy"
  )
  class(atlas) <- c("toy", "surfatlas", "atlas")
  parent <- surface_space(
    vertex_ids = c(paste0("L-", 1:3), paste0("R-", 1:3)),
    hemisphere = rep(c("left", "right"), each = 3),
    topology = rbind(c(1, 2, 3), c(4, 5, 6)),
    geometry = rbind(left, right),
    template = "toy"
  )
  x <- parcel_space_from_atlas(atlas, parent)
  feature_ids(x)
  if (is.na(old_rgl)) Sys.unsetenv("RGL_USE_NULL") else Sys.setenv(RGL_USE_NULL = old_rgl)
}
# }
```
