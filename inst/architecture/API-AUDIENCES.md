# API audiences

`fmridataset` exposes one package namespace with three documented audiences.
This classification is normative for 1.0 review and generated namespace tests.

## User API

The ordinary user surface covers:

- frame, collection, and study construction;
- observation, feature, entity, relation, assay, and space inspection;
- lazy filtering, feature mapping, binding, and validity policies;
- bounded numerical and spatial realization;
- NIfTI, HDF5, Zarr, atlas, surface, and latent interoperability;
- FDS inspection, persistence, and explicit legacy migration.

## Extension API

Backend, spatial, mapping, relation, and codec implementers may rely on:

- `ArraySource` generics, descriptors, validators, and lifecycle rules;
- `FeatureSpace` generics and compatibility laws;
- the metadata-only `frame_schema()` contract and mode-specific schema
  validation used by collections, binding, FDS codecs, and consumers;
- feature-map, provenance, entity, relation, validity, and hierarchy contracts;
- FDS manifests, bindings, validators, digests, and reconstruction helpers;
- block planning and source composition required by storage implementations.

An extension object is a serializable descriptor. Open file handles and caches
are runtime products of `source_open()` and must never become semantic state.
Methods must preserve requested order, reject ambiguous identity, and return
non-dropping two-dimensional blocks.

The stable extension groups are:

- source protocol: `as_array_source()`, `source_descriptor()`,
  `validate_array_source()`, and the `source_*()` lifecycle generics;
- spatial protocol: `n_features()`, `feature_ids()`, `native_shape()`,
  `restrict_space()`, `vectorize_space()`, `reconstruct_space()`,
  `adjacency()`, exact `same_space()`, the migration alias
  `compatible_space()`, and `space_digest()`;
- typed `identity_descriptor()` results and the versioned R-only
  `canonicalization_contract()`;
- mapping and provenance: `feature_map*()`, `provenance_*()`, and their
  validators and digests;
- semantic registries: entity, relation, hierarchy, validity, and mask-bank
  constructors, validators, and digests;
- storage protocol: FDS manifests, bindings, validation, reconstruction, and
  source-composition/block-planning helpers.

Constructor and validator laws are executable in
`tests/testthat/helper-frame-conformance.R`, `test-frame-properties.R`,
`test-array-source.R`, `test-feature-space.R`, `test-feature-map.R`, and the
FDS tests. Downstream protocol checks run against `fmristore`, `multidesign`,
and `fmrigds` before an extension-surface change is accepted.

## Developer-only API

`counting_source()`, `source_counts()`, `reset_source_counts()`, and
`fault_source()` remain exported so companion packages can certify zero-I/O,
lifecycle, atomic-write, and failure-cleanup laws. They are test instruments,
not application storage or provenance.

Synthetic vignette helpers, runtime handle classes, internal registries,
codec internals, error constructors, and `%||%` are not public API.
