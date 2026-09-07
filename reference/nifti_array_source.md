# Construct a pushdown-aware NIfTI array source

The descriptor reads headers and one mask at construction, but no fMRI
volumes. Numerical reads split requested global observations by file,
pass local volume indices into
[`neuroim2::read_vec()`](https://bbuchsbaum.github.io/neuroim2/reference/read_vec.html),
and restrict the mask to requested packed features before
materialization. Native reads return full-volume `NeuroVec` objects in
requested observation order.

## Usage

``` r
nifti_array_source(paths, mask, chunks = NULL)
```

## Arguments

- paths:

  One or more NIfTI files with a common spatial grid.

- mask:

  A NIfTI mask path or a compatible `volume_space`.

- chunks:

  Optional logical observation-by-feature chunk hint.

## Value

A serializable `nifti_array_source`.

## Details

The fingerprint covers the descriptor and the size and modification time
of every file captured at construction, never the voxel values. Every
open, read, and native read re-observes those files first and raises
`fmridataset_error_source_stale` (with `source`, `expected`, `actual`,
and `changed` fields) if any differ; genuine read failures remain
`fmridataset_error_backend_io`. See
[`content_hash()`](https://bbuchsbaum.github.io/fmridataset/reference/content_hash.md)
to identify values.

## Examples

``` r
# A small NIfTI fixture shipped with neuroim2 stands in for real data.
path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
if (nzchar(path)) {
  src <- nifti_array_source(path, path)
  source_shape(src)
  source_dtype(src)
}
#> [1] "float32"
```
