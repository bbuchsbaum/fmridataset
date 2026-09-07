# Feature-space contract

Every feature-space class (such as
[`volume_space()`](https://bbuchsbaum.github.io/fmridataset/reference/volume_space.md),
[`surface_space()`](https://bbuchsbaum.github.io/fmridataset/reference/surface_space.md),
[`parcel_space()`](https://bbuchsbaum.github.io/fmridataset/reference/parcel_space.md),
[`basis_space()`](https://bbuchsbaum.github.io/fmridataset/reference/basis_space.md),
[`composite_space()`](https://bbuchsbaum.github.io/fmridataset/reference/composite_space.md),
and
[`index_space()`](https://bbuchsbaum.github.io/fmridataset/reference/index_space.md))
implements this shared generic contract for feature count, stable
identity, restriction, vectorization, and reconstruction.

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

same_space(x, y, ...)

assert_same_space(x, y, ...)

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

## Value

`n_features()` returns a single integer; `feature_ids()` returns a
character vector of stable feature identifiers; `native_shape()` returns
the class-specific native dimensions; `feature_data()` returns a tibble
of per-feature metadata; `space_digest()` returns a content digest
string; `restrict_space()` and `reconstruct_space()` return an object of
the same feature-space class; `vectorize_space()` returns a numeric
vector; `adjacency()` returns a sparse adjacency matrix or `NULL`;
`same_space()` and `compatible_space()` return a `space_compatibility`
list; `assert_same_space()` and `assert_compatible_space()` return the
compatibility report invisibly and signal an error when incompatible.

## Examples

``` r
x <- index_space(3, ids = c("a", "b", "c"))
n_features(x)
#> [1] 3
feature_ids(x)
#> [1] "a" "b" "c"
y <- restrict_space(x, 1:2)
same_space(x, x)$same
#> [1] TRUE
```
