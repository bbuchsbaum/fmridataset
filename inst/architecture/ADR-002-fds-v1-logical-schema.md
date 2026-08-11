# ADR-002: FDS logical schema version 1

Status: accepted
Date: 2026-08-11

## Decision

FDS version 1 freezes the backend-neutral semantic manifest for an
`fmri_frame`. Its identity is `org.fmridataset.fds/v1`, with integer version
`1`. The normative executable contract is `fds_frame_manifest()` plus
`validate_fds_manifest()`; `inst/schema/fds-v1.schema.json` describes the same
top-level envelope for non-R codecs.

The required manifest fields are:

- schema identity, object type, and logical shape;
- observation and feature IDs, scalar tables, aligned blocks, and axis
  metadata;
- the immutable feature space on the feature axis;
- a named-axis array registry covering assays and multivariate axis blocks;
- source-free assay descriptors including dtype, shape, roles, units, and
  exact axis digests;
- entities, relations, auxiliary tables, active assay, metadata, provenance,
  and a named extension registry.

Physical sources are bound separately by assay name. URIs, HDF5 dataset names,
chunk grids, compression, checksums, caches, handles, and backend fingerprints
are codec state and are not part of the logical manifest. Consequently the
same frame has the same FDS manifest when rechunked or moved between HDF5,
Zarr, memory, and sharded sources.

## Compatibility rules

1. A version-1 reader accepts schema ID `org.fmridataset.fds/v1` and integer
   version `1` only.
2. Older or future major versions require an explicit, tested migration before
   frame construction. Readers never guess compatibility from similar fields.
3. Required field meanings cannot change within version 1. Additive data must
   live under `extensions` with a stable, namespaced key.
4. Codecs must preserve IDs, row order, factor levels and ordering, missingness,
   component metadata, feature-space identity, and provenance exactly.
5. Codecs validate the semantic manifest before binding physical sources, then
   validate each bound source's name, shape, and dtype before returning a
   frame.
6. Canonical manifests contain no functions, environments, external pointers,
   open handles, or other runtime state.

## Consequences

`fmridataset` owns manifest construction, validation, digesting, and frame
reconstruction. `fmristore` and future codecs own physical layouts and atomic
commit protocols. Migrations are pure semantic transformations with golden
fixtures; they are not hidden inside backend readers.
