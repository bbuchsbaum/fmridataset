# Feature-space contract

Feature-space contract

## Usage

``` r
n_features(x, ...)

feature_ids(x, ...)

native_shape(x, ...)

feature_data(x, ...)

space_digest(x, ...)

restrict_space(x, index, ...)

vectorize_space(x, spatial_object, ...)

reconstruct_space(x, vector, ...)

adjacency(x, ...)

compatible_space(x, y, ...)

assert_compatible_space(x, y, ...)

# S3 method for class 'surface_space'
reconstruct_space(x, vector, format = c("surface_map", "neurosurf"), ...)
```

## Arguments

- x:

  A feature-space object.

- ...:

  Additional arguments for methods.

- index:

  Feature positions used to restrict a space.

- spatial_object:

  A native spatial object to vectorize.

- vector:

  A feature vector to reconstruct.

- y:

  A second feature-space object.

- format:

  Surface reconstruction format. The backend-neutral default is
  `"surface_map"`; `"neurosurf"` returns a
  [`neurosurf::NeuroSurface`](https://bbuchsbaum.github.io/neurosurf/reference/NeuroSurface.html)
  when embedded unilateral geometry is available.
