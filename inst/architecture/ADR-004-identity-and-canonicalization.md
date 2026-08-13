# ADR-004: Identity domains and canonicalization

Status: accepted for the 1.0 contract.

## Decision

Identity is typed. Equal hexadecimal strings from different domains are not
interchangeable.

| Domain | Includes | Excludes | Public operation |
|---|---|---|---|
| Semantic | Source-free frame/study manifest, axis IDs, annotations, entities, relations, spaces, provenance | Physical paths, handles, chunks, compression | `fds_manifest_digest()`, `study_digest()`, `collection_digest()` |
| Schema | Types, factor levels, assay annotations, block trailing axes/components, space identity, relation/table contracts | Observation count in bind/collection projections; numerical values | `frame_schema_digest()` |
| Space | Exact feature IDs, ordered support, geometry/topology or parent/operator identity, declared metadata | Mere dimension or feature count | `space_digest()` and `same_space()` |
| Source | Serializable physical descriptor, selector/shard composition, relevant physical metadata | Semantic annotations and uncomputed contents | `source_fingerprint()` |
| Provenance | Typed derivation records, inputs, outputs, software and parameters | Runtime handles and caches | `provenance_digest()` |
| Content | Bytes or normalized numerical values under a declared external procedure | Assumed equality from source identity | Optional backend receipt passed to `identity_descriptor()` |

Canonicalization v1 is `org.fmridataset.r-canonical/v1`: SHA-256 over a
package-owned tagged binary encoding. Lengths and numeric payloads use
big-endian byte order; named fields and attributes are ordered
lexicographically; strings are UTF-8 NFC; sparse matrices use canonical
column-major triplets; and missing, null, NA, NaN, infinities, and negative
zero remain explicitly distinguished. It is R-only because no non-R reference
decoder is part of v1. Cross-language identity requires a future specified
decoder and a new canonicalization version; v1 digests must never be relabeled
as cross-language hashes.

Physical relocation changes source identity when the source descriptor changes,
but not semantic, schema, or space identity. A storage codec may attach a
content receipt without reading data during ordinary inspection.

`same_space()` means exact semantic spatial identity: class, digest, and ordered
feature IDs all agree. Equal dimensions or feature counts are never evidence of
identity. `compatible_space()` and `assert_compatible_space()` remain migration
aliases for exact identity during the 1.0 transition; the term “compatible” does
not establish the existence of a map. A relationship between different spaces
requires an explicit validated `feature_map`.

## Consequences

- compatibility code compares typed identities, never shapes alone;
- `explain()` reports schema, semantic, space, and source identities separately;
- FDS manifests remain independent of physical locations;
- content hashing is opt-in and backend-specific because it may require I/O;
- changing canonical encoding requires an explicit version and golden vectors.
