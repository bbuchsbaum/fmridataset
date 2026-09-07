# Lazily transform a frame into a new feature domain

Lazily transform a frame into a new feature domain

## Usage

``` r
map_features(x, target = NULL, map = NULL, assay_rules = "linear")
```

## Arguments

- x:

  An `fmri_frame` or view.

- target:

  Optional parent-linked target space from which a canonical map can be
  derived.

- map:

  Optional explicit `feature_map`.

- assay_rules:

  Named rules for every assay: `"linear"` or `"independent_variance"`.
  Unnamed scalar rules are recycled.

## Value

A new linked-domain `fmri_frame` whose assays remain lazy.

## Examples

``` r
parent <- volume_space(c(2, 2, 1), support = 1:4, template = "toy")
frame <- fmri_frame(
  assays = list(signal = matrix(1:12, nrow = 3)),
  observations = data.frame(.obs_id = paste0("o", 1:3)),
  space = parent
)
parcels <- parcel_space(
  parent,
  parcel_ids = c("left", "right"),
  membership = Matrix::sparseMatrix(
    i = 1:4, j = c(1L, 1L, 2L, 2L), x = 1, dims = c(4L, 2L)
  ),
  atlas = "toy-atlas"
)
parcel_frame <- map_features(frame, target = parcels)
dim(collect_assay(parcel_frame))
#> [1] 3 2
```
