# Open a BIDS HDF5 Study Archive

Opens a BIDS HDF5 archive created by
[`compress_bids_study()`](https://bbuchsbaum.github.io/fmridataset/reference/compress_bids_study.md)
and returns a `bids_h5_study_dataset` that is a subclass of
`fmri_study_dataset`. All standard fmridataset methods
(`get_data_matrix`, `data_chunks`, `as_delarr`, etc.) work on the
returned object.

## Usage

``` r
bids_h5_dataset(file, preload = FALSE)
```

## Arguments

- file:

  Character string. Path to the `.h5` BIDS archive.

- preload:

  Logical. Reserved for future use (ignored in Phase 1).

## Value

A `bids_h5_study_dataset` object (subclass of `fmri_study_dataset`).

## See also

[`compress_bids_study`](https://bbuchsbaum.github.io/fmridataset/reference/compress_bids_study.md),
[`subset_bids_h5`](https://bbuchsbaum.github.io/fmridataset/reference/subset_bids_h5.md),
[`participants`](https://bbuchsbaum.github.io/fmridataset/reference/participants.md),
[`tasks`](https://bbuchsbaum.github.io/fmridataset/reference/tasks.md),
[`sessions`](https://bbuchsbaum.github.io/fmridataset/reference/sessions.md)
