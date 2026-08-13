# BIDS H5 Dataset Reader

Opens a BIDS HDF5 archive written by
[`compress_bids_study()`](https://bbuchsbaum.github.io/fmridataset/reference/compress_bids_study.md)
and returns a `bids_h5_study_dataset` object that is a subclass of
`fmri_study_dataset`. The study-level object exposes the full
`fmridataset` API (data_chunks, as_delarr, get_data_matrix, etc.)
together with BIDS-specific accessors for participants, tasks, sessions,
the scan manifest, parcellation metadata, and confound regressors.

## Details

Internally the reader:

1.  Opens the H5 file via a shared, ref-counted connection.

2.  Reads the `/scan_index/` table to build the scan manifest.

3.  Creates one lightweight `bids_h5_scan_backend` per scan.

4.  Groups scan backends by subject; multi-run subjects get a nested
    `study_backend` over their scan backends.

5.  Composes per-subject `fmri_dataset` objects into a
    `fmri_study_dataset` via
    [`fmri_study_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_study_dataset.md).

6.  Wraps the result as a `bids_h5_study_dataset` with the scan
    manifest, shared H5 connection, and a flat list of per-scan backends
    (used by
    [`subset_bids_h5()`](https://bbuchsbaum.github.io/fmridataset/reference/subset_bids_h5.md)).

Parcellated data lives in feature-space (K parcel columns). ROI/sphere/
voxel selectors do not apply; use
[`index_selector()`](https://bbuchsbaum.github.io/fmridataset/reference/index_selector.md)
to select parcels by column index.
