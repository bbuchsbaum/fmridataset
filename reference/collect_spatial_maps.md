# Collect spatial maps through native or reconstructed reads

Collect spatial maps through native or reconstructed reads

## Usage

``` r
collect_spatial_maps(
  x,
  observations = NULL,
  assay = active_assay(x),
  path = c("auto", "native", "reconstruct"),
  memory_budget = getOption("fmridataset.spatial_budget", 512 * 1024^2)
)
```

## Arguments

- x:

  An `fmri_frame` or view.

- observations:

  Observation IDs or integer positions. The requested order is
  preserved. Duplicated selectors are rejected, as they are on every
  other frame axis selection.

- assay:

  Assay name.

- path:

  One of `"auto"`, `"native"`, or `"reconstruct"`.

- memory_budget:

  Maximum estimated peak bytes for all returned native maps plus the
  current packed read, conversion, and reconstruction buffers.

## Value

A named list with one native spatial object per observation.

## Examples

``` r
sp <- volume_space(dim = c(2L, 2L, 1L), affine = diag(4), support = 1:4)
frame <- fmri_frame(
  assays = list(signal = memory_source(matrix(seq_len(12), 3, 4))),
  observations = data.frame(.obs_id = sprintf("obs-%d", 1:3)),
  space = sp,
  active_assay = "signal"
)
maps <- collect_spatial_maps(frame, observations = c(1L, 2L))
names(maps)
#> [1] "obs-1" "obs-2"
```
