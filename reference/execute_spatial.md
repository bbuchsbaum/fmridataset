# Stream an operation over spatial maps

Unlike
[`collect_spatial_maps()`](https://bbuchsbaum.github.io/fmridataset/reference/collect_spatial_maps.md),
`execute_spatial()` holds only one input map at a time. The callback
result is retained, so callers remain responsible for keeping returned
values appropriately small.

## Usage

``` r
execute_spatial(
  x,
  observations = NULL,
  FUN,
  ...,
  assay = active_assay(x),
  path = c("auto", "native", "reconstruct"),
  memory_budget = getOption("fmridataset.spatial_budget", 512 * 1024^2)
)
```

## Arguments

- x:

  An `fmri_frame` or view.

- observations:

  Observation IDs or integer positions.

- FUN:

  Function receiving `map` and `observation_id`.

- ...:

  Additional arguments passed to `FUN`.

- assay:

  Assay name.

- path:

  One of `"auto"`, `"native"`, or `"reconstruct"`.

- memory_budget:

  Maximum estimated peak bytes for one input spatial map plus its packed
  read, conversion, and reconstruction buffers.

## Value

A list of callback results in requested observation order.

## Examples

``` r
sp <- volume_space(dim = c(2L, 2L, 1L), affine = diag(4), support = 1:4)
frame <- fmri_frame(
  assays = list(signal = memory_source(matrix(seq_len(12), 3, 4))),
  observations = data.frame(.obs_id = sprintf("obs-%d", 1:3)),
  space = sp,
  active_assay = "signal"
)
execute_spatial(frame, c(1L, 2L), function(map, observation_id) observation_id)
#> [[1]]
#> [1] "obs-1"
#> 
#> [[2]]
#> [1] "obs-2"
#> 
```
