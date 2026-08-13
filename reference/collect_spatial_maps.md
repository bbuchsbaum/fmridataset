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

  Observation IDs or integer positions. The requested order and
  duplicates are preserved.

- assay:

  Assay name.

- path:

  One of `"auto"`, `"native"`, or `"reconstruct"`.

- memory_budget:

  Maximum estimated bytes for all returned native maps.

## Value

A named list with one native spatial object per observation.
