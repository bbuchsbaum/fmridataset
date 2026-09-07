# ADR-009: Source fingerprints and content hashes

Status: accepted for the 1.0 contract. Refines the source and content rows of
ADR-006.

## Decision

A source has two separately computed identities and the package never
substitutes one for the other.

| Identity | Operation | Hashes | Cost | When computed |
|---|---|---|---|---|
| Revision fingerprint | `source_fingerprint()` | The serializable descriptor: type, schema version, shape, dtype, chunk grid, selectors, shard composition, and the physical revision evidence the backend can observe without reading values (file sizes and mtimes, store metadata, a per-object identity token) | O(descriptor): O(1) for a stored descriptor, O(selector length) once for a view | Once, at construction; cached in the descriptor |
| Content hash | `content_hash()` | The realized numerical values in row-major order, plus shape and realized R mode | O(n) in values, streamed through bounded reads | Only on request |

Semantic, schema, space, and provenance identity are unchanged from ADR-006
and include neither of these.

### Revision fingerprints

`source_fingerprint()` answers "is this the same descriptor of the same
revision of the same physical thing". It never reads array values.

- **File-backed sources** (NIfTI, Zarr) derive the fingerprint from the
  descriptor and the physical metadata captured at construction: file paths,
  sizes, and mtimes for NIfTI; URI, array path, shape, chunks, and dtype for
  Zarr. Every open, read, and native read first re-observes that metadata and
  compares it with the descriptor.
- **Memory sources** hold their values in the descriptor and have no file
  state to observe. Their fingerprint is derived from shape, dtype, chunks,
  a per-object identity token (a UUID generated once at construction), and an
  optional caller-supplied `revision` string. Consequently two independently
  constructed memory sources with equal values have different fingerprints by
  design, a serialization round trip preserves the fingerprint, and
  fingerprint equality is never value equality. A caller who wants
  equal-valued memory sources to agree opts in with
  `memory_source(identity = "content")`, which hashes the payload once at
  construction and pays the O(n) cost visibly.
- **Sparse entity sources** wrap small immutable metadata matrices. Their
  fingerprint is content-derived, computed exactly once at construction, and
  cached, so lifted entity blocks are deterministic across resolutions without
  re-hashing on every call.
- **Wrappers** (views, row-index lifts, row-bound and row-sharded
  compositions, feature maps, validity masks, instrumentation) combine the
  cached fingerprints of their children with their own selectors. They never
  re-hash a child.

Fingerprint construction is therefore free of payload reads for every source,
and hot paths that fingerprint on every call (`plan_blocks()`,
`execute_block_plan()`, `explain()`, collection descriptors, provenance
inputs) are O(descriptor).

### Content hashes

`content_hash()` answers "do these arrays hold the same values". It is the
only operation in the package that identifies an array by its values and it is
always explicit. Nothing computes it implicitly: construction, validation,
inspection, `explain()`, planning, FDS manifests, `write_frame()`, and
`open_frame()` never call it.

Content hashing v1 (`org.fmridataset.content-hash/v1`) is a chained SHA-256
over fixed-size leaves of 65 536 realized values in row-major order, preceded
by a header of the contract identifier, the two-dimensional shape, and the
realized R mode (double, logical, or complex). Realization follows the source
dtype exactly as reads do: every numeric storage dtype realizes as double.
NaN and NA are each normalized to one representation and remain distinct;
negative zero is preserved. The digest is independent of storage dtype, chunk
grid, read block size, wrappers, views, and shard composition, so a memory
copy, a chunked file, and a row-bound composition of the same values agree.

The default method opens one handle, reads row blocks aligned to the source
chunk grid and bounded by `block_bytes`, and folds each block into the running
digest; no source is materialized whole. Row-bound sources stream shard by
shard so a read never crosses a shard boundary. Memory sources hash in the
same bounded blocks rather than transposing the whole payload.

The result is a receipt for the values at the moment of reading. It is
accepted by `identity_descriptor(domain = "content", content_digest = ...)`
and may be recorded in provenance or metadata by whoever computed it. The
FDS manifest schema does not carry it.

### Caching and lifecycle

Descriptors are immutable values. Fingerprints are computed at construction
and stored in the descriptor; the accessor returns the stored value.
Serialization preserves them. Mutating a descriptor in place is outside the
contract and is not detected, with one deliberate exception: in-memory
mutation of a memory source's payload leaves its fingerprint unchanged, and
shows only when a fresh `content_hash()` is compared with an earlier one.

Content hashes are not cached in descriptors. A cached hash would be a claim
about values the package has not re-read.

### Failure behavior

Two error classes are distinct and callers may rely on the distinction:

- `fmridataset_error_source_stale`: the physical thing a descriptor describes
  has changed since construction. Raised by every file-backed source before any
  numerical read, with fields `source` (type and location), `expected` (the
  metadata captured at construction), and `actual` (the metadata observed now).
  The correct response is to rebuild the descriptor, not to retry.
- `fmridataset_error_backend_io`: the backend could not perform an operation
  (missing package, unreadable file, closed handle, injected fault). The
  descriptor may still be current.

`content_hash()` raises `fmridataset_error_source_contract` if a source
delivers a different number of values than its shape declares, and passes
through stale and I/O errors from the reads it issues.

## How to choose

- Detecting that a file or store changed under a descriptor: rely on the
  automatic freshness checks and catch `fmridataset_error_source_stale`.
- Keying a cache, a plan, or a provenance input on "the same source in the
  same revision": `source_fingerprint()`.
- Deciding whether two arrays hold the same values, certifying a copy, or
  attaching a value receipt to a persisted or published artifact:
  `content_hash()`, called explicitly where the O(n) read is acceptable.
- Wanting equal-valued memory sources to share a fingerprint:
  `memory_source(identity = "content")` (the token is derived from the
  values), or pass the same `revision` to each (the revision replaces the
  per-object token, so the caller is asserting that the sources are the same
  source in the same revision).
- Never infer value equality from any fingerprint, and never infer that a
  content hash is current without recomputing it.

## Consequences

- Constructing a memory source of any size hashes no payload bytes.
- Memory-source fingerprints are per object; consumers that compared
  fingerprints across independently constructed equal matrices must compare
  content hashes instead.
- `fds_manifest_digest()` is source-free: two frames that differ only in
  the identity tokens of their memory sources, including an entity block
  supplied as a `memory_source` rather than a matrix, have equal manifest
  digests. `entity_registry_digest()` canonicalizes the registry object
  itself, so it does include such a block's identity token and is per object;
  `bind_observations()` therefore compares registries semantically (names,
  keys, scalar data, block components, and block values, by fingerprint first
  and by realized values only when fingerprints differ) rather than by that
  digest. Supply matrices for entity blocks whose registry digest must agree
  across independently built frames, or compare content hashes.
- NIfTI stale detection changed class from `fmridataset_error_backend_io` to
  `fmridataset_error_source_stale`.
- `content_hash()` joins the extension API; `content_hash_contract()`
  publishes its versioned rules alongside `canonicalization_contract()`.
