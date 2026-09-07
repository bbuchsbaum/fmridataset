# Construct a collection of semantically equivalent fMRI frames

A collection keeps frames separate when they share an observational and
assay contract but cannot share a feature axis, as with
participant-native volume or surface spaces. Equal feature dimensions or
IDs are not required; feature-space type and annotation semantics are
validated explicitly.

## Usage

``` r
fmri_collection(frames, metadata = list(), provenance = NULL)
```

## Arguments

- frames:

  A non-empty named list of `fmri_frame` objects or lazy views.

- metadata:

  Unaligned collection-level metadata.

- provenance:

  `NULL` or a validated `provenance_graph`.

## Value

An `fmri_collection`.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
collection <- fmri_collection(list(sub01 = frame, sub02 = frame))
collection_ids(collection)
#> [1] "sub01" "sub02"
```
