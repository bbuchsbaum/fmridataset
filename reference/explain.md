# Explain a frame without reading numerical values

`explain()` returns a bounded, serializable summary of the visible
frame. Axis IDs are sampled by default so inspecting a large study does
not create another large object. Set `ids = "complete"` only when every
visible ID is required.

## Usage

``` r
explain(x, ids = c("sample", "none", "complete"), sample_size = 3L)
```

## Arguments

- x:

  An `fmri_frame` or view.

- ids:

  One of `"sample"`, `"none"`, or `"complete"`.

- sample_size:

  Number of IDs sampled from each end of each axis.

## Value

A bounded serializable execution summary. No assay or aligned-block
values are read. `ids_durable` reports whether the observation axis and
the feature space carry durable IDs; when it is `FALSE` the semantic
digest is `NULL` because ephemeral IDs cannot enter an FDS manifest.

## Details

The schema digest describes column, block, assay, relation, entity,
table, and space contracts. The semantic digest covers the complete
source-free FDS manifest. Physical source fingerprints are reported
separately.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
summary <- explain(frame)
summary$shape
#> observation     feature 
#>           4           8 
```
