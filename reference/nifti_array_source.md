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
