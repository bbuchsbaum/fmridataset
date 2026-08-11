# ADR-001: Canonical semantic data model

Status: accepted
Date: 2026-08-11

## Decision

`fmridataset` owns the canonical semantic representation for fMRI data:

- `fmri_frame` represents one observation domain by one feature domain;
- `fmri_collection` groups equivalent frames that cannot yet share a feature
  axis, such as native-space participant data;
- `fmri_study` links heterogeneous frames through shared entities, relations,
  mappings, and provenance.

Every frame assay has the same observation IDs and feature IDs. Factors and
continuous variables are annotations rather than array dimensions.
Multivariate annotations are explicit axis blocks. Spatial compatibility is
established by feature IDs and a feature-space digest, never by dimensions
alone.

Canonical objects contain serializable descriptors, not open file handles,
environments, external pointers, or arbitrary loader functions. Runtime array
execution is constructed from those descriptors when needed.

## Package boundaries

| Package | Responsibility |
|---|---|
| `fmridataset` | Frames, studies, axes, IDs, entities, relations, spaces, sources, views, and the logical FDS schema |
| `delarr` | Lazy numerical plans, fusion, block execution, and bounded realization |
| `fmristore` | Certified HDF5 codecs, layouts, atomic writes, append, and recovery |
| `multidesign` | Formula parsing, multivariate terms, contrasts, folds, and compiled designs |
| `fmrigds` | Statistical plans and feature-block kernels producing result frames |
| `neuroim2` | In-memory volume objects and native spatial interoperability |

Shared semantic generics are defined by `fmridataset`; companion packages
register methods rather than redefining the concepts.

## Compatibility and releases

- Development begins at 0.10.0. The divergent historical `v0.9.0` tag is not
  rewritten.
- Legacy constructors and classes remain migration adapters throughout the
  0.x transition and are removed from the 1.0 public API.
- HDF5 is the certified 1.0 persistent backend. Zarr remains experimental
  until it independently passes the same conformance gates.
- `DelayedArray` is optional interoperability, not an internal execution path.

## Consequences

Raw time series, beta estimates, parcel data, surfaces, and latent
representations use the same two-axis contract but remain separate linked
frames when either logical domain changes. Native-space participants remain a
collection until an explicit map creates a common feature space. Statistical
model matrices are derived and reproducibly described; they are not canonical
dataset state.
