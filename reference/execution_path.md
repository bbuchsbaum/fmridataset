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

## Examples

``` r
sp <- volume_space(dim = c(2L, 2L, 1L), affine = diag(4), support = 1:4)
frame <- fmri_frame(
  assays = list(signal = memory_source(matrix(seq_len(12), 3, 4))),
  observations = data.frame(.obs_id = sprintf("obs-%d", 1:3)),
  space = sp,
  active_assay = "signal"
)
execution_path(frame, operation = "matrix")
#> [1] "matrix"
execution_path(frame, operation = "spatial")
#> [1] "reconstruct"
```
