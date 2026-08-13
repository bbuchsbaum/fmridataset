# Select a matrix or spatial execution path

Matrix operations always use bounded observation-by-feature blocks.
Spatial operations use a source's native-image capability only when the
frame view retains the complete feature domain; otherwise they
reconstruct maps from packed assay values through the frame's feature
space.

## Usage

``` r
execution_path(
  x,
  operation = c("matrix", "spatial"),
  assay = active_assay(x),
  path = c("auto", "native", "reconstruct")
)
```

## Arguments

- x:

  An `fmri_frame` or view.

- operation:

  Either `"matrix"` or `"spatial"`.

- assay:

  Assay name.

- path:

  For spatial operations, one of `"auto"`, `"native"`, or
  `"reconstruct"`.

## Value

One of `"matrix"`, `"native"`, or `"reconstruct"`.
