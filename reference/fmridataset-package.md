# fmridataset: spatially typed, annotated fMRI datasets

`fmridataset` keeps fMRI arrays aligned with what their rows and columns
mean. Its canonical container,
[`fmri_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_frame.md),
represents one observation domain by one feature domain and carries
stable IDs, annotations, entities, relations, and explicit spatial
identity through lazy views, feature maps, binding, and storage round
trips.
[`fmri_collection()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_collection.md)
groups equivalent frames that cannot yet share a feature axis, and
[`fmri_study()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_study.md)
links heterogeneous frames through shared entities and typed links.

## Details

Numerical values live behind serializable array sources
([`memory_source()`](https://bbuchsbaum.github.io/fmridataset/reference/memory_source.md),
[`nifti_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/nifti_array_source.md),
[`zarr_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/zarr_array_source.md),
and the sources provided by storage packages) and are read only on
request, under a realization budget. The logical frame contract is
persisted through FDS manifests
([`fds_frame_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/fds_frame_manifest.md))
and HDF5 via
[`write_frame()`](https://bbuchsbaum.github.io/fmridataset/reference/write_frame.md).

See `inst/architecture/` for the decision records that define the data
model, the FDS schema, and the identity, temporal, and ID policies.

## See also

Useful links:

- <https://github.com/bbuchsbaum/fmridataset>

- <https://bbuchsbaum.github.io/fmridataset/>

- Report bugs at <https://github.com/bbuchsbaum/fmridataset/issues>

## Author

**Maintainer**: Bradley Buchsbaum <bbuchsbaum@gmail.com>
([ORCID](https://orcid.org/0000-0001-5800-9890))
