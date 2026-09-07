# ADR-010: Selection algebra

Status: accepted for the 1.0 contract. Refines the selector rows of ADR-006
and ADR-009.

## Context

Frames, views, source views, collections, axis and entity frames, and study
filters each normalized selectors independently, and disagreed: frames and
collections rejected repeated elements while raw sources accepted them,
collections rejected negative positions that frames allowed, and a raw source
could not tell a character selector from a mistake. Every select-all axis was
spelled out as `seq_len(n)`: a view of a large frame stored two full index
vectors in its assay descriptors, fingerprinted them in O(n), and re-derived
chunk runs on every Zarr read. Nested source views kept one index vector per
level.

## Decision

### One internal type with three canonical forms

Every axis selection is an internal `axis_selection` value (`R/axis-selection.R`)
with the axis length `n` and one of three forms:

| Form | Stores | Meaning |
|---|---|---|
| `all` | nothing | every element of the axis, in order |
| `range` | `start`, `end` | one contiguous ascending run |
| `positions` | an integer vector | an explicit unique, order-preserving subset; also the empty selection on a non-empty axis |

Construction canonicalizes: positions that spell out the whole axis become
`all`, and positions that form one ascending run become `range`. On an
axis of length zero the empty vector spells out the whole axis, so every
selection over an empty axis is `all`; `NULL`, `integer()`, and
`logical()` on such an axis produce one descriptor, and a plan built over an
empty frame matches the same frame subset with `integer()`. Equal
selections therefore have equal descriptors however a caller expressed them,
and a descriptor's size never scales with the axis length unless the caller
actually enumerated an arbitrary subset.

The persisted form is always positional. Stable IDs are resolved against the
owning axis at normalization time: frames, views, collections, axis frames,
and entity frames own IDs and accept them; raw sources are positional and
refuse character selectors with `reason = "positional_axis"`.

No bitmap form was added. In R a logical mask costs four bytes per axis
element, the same as an integer position, and there is no packed bitset
without a dependency; a mask therefore never beats `positions` in size and
loses the request order that positions carry. Logical selectors normalize to
positions. A packed form can be added later as a fourth form without changing
the law.

### One normalization law

`.normalize_selection(index, n, ids, axis, abort)` is the only path from a
caller's selector to an `axis_selection`, and every owner calls it:

- `[.fmri_frame`, `[.fmri_view`, `spatial_map()`, `collect_spatial_maps()`,
  `execute_spatial()` (frame axes);
- `source_view()`, `source_read()`, `source_read_native()`,
  `source_realization_cost()`, `locate_source_rows()`, and every built-in
  source's read method (positional axes);
- `[.fmri_collection` and `[[.fmri_collection`;
- `[.axis_frame` and `[.entity_frame`, through which `filter_entities()`
  restricts the study's entity registry by the kept entity IDs.

The law:

- character selectors are stable IDs; each must exist, must be unique, and
  keeps its request order;
- logical selectors must match the axis length and contain no `NA`;
- numeric selectors must be finite whole numbers within R's integer range,
  may reorder, may be negative but must not mix signs, drop zero, and must be
  in bounds (negative positions past the axis are an error, not silently
  ignored; `Inf`, `-Inf`, and magnitudes beyond `.Machine$integer.max` are
  rejected before any coercion);
- an element appears at most once: repeated positions or IDs are an error
  everywhere, on raw sources exactly as on frames;
- an empty selection is a legal zero-length axis everywhere.

Errors carry the owner's class (`fmridataset_error_alignment` for frames,
views, sources, and axes; `fmridataset_error_collection` for collections) and
a structured `reason`: `unsupported_type`, `positional_axis`, `missing`,
`duplicate`, `unknown_id`, `length`, `non_finite`, `non_integer`,
`mixed_sign`, `out_of_bounds`, or `axis_length`.

The law is enforced for sources at the `source_read()` and
`source_read_native()` generics, before dispatch, so extension sources that
the package does not implement reject the same selectors the same way. Methods
still receive the caller's `NULL`, integer, or logical selector and expand it
themselves; a method that also normalizes is redundant, not wrong.

### The two open questions

**Duplicates.** Rejected everywhere. The frame contract says an axis element
appears once, and a source is what a frame reads through, so a source that
repeated rows would let the positional and the ID-bearing layers disagree
about the same read. Callers that need a repeated row gather it after the
read. This changed the raw-source contract: the conformance gate that pinned
"repeated positions select repeated data" now pins the rejection.

**Empty.** Allowed everywhere as a selector. Frames already plan zero blocks
and collect zero-extent matrices for an emptied axis, and a source view with
an empty axis is a legal zero-by-`m` array. A collection still cannot be
empty, but that is a container invariant, raised after normalization by
`[.fmri_collection` with the collection's own error class and
`reason = "empty_collection"`; it is not a rule of the selector algebra.

### Composition

`.selection_compose(outer, inner)` resolves a selection applied to the axis an
inner selection presents into one selection over the axis the inner one
selects from. `all` on either side is the identity, a range over a range is a
range, and only a `positions` result touches a vector. Consequently:

- `source_view()` over a `source_view` composes into one view over the root
  source; nested views never accumulate;
- `[.fmri_view` composes into one view over the base frame;
- `plan_blocks()` and `execute_block_plan()` slice a frame per block by
  composing a range over the frame's selection, which stays a range;
- Zarr reads take their chunk-aligned runs directly from the form: `all` and
  `range` are one run, and only arbitrary positions are decomposed.

### Descriptors and identity

`source_view` stores the two normalized selections and a `schema_version` of
2. Its revision fingerprint (ADR-009) hashes the canonical form, so a
select-all or range view fingerprints in constant time and equal selections
agree whether they were expressed by ID, mask, positions, negation, or
nesting. `plan_blocks()` hashes the same form. Neither the FDS manifest nor
`fds_manifest_digest()` changed: manifests are source-free and never carried
selectors.

`explain()` reports each axis's form, count, and axis length (and range
bounds) under `selection`, never the positions vector, so it remains bounded.

### Pushdown declarations

A source declares the selector forms it consumes natively as capability
strings `pushdown:all`, `pushdown:range`, and `pushdown:positions`, inspectable
through `source_capabilities()`; forms it does not declare are emulated by
the source itself. The built-in declarations are:

| Source | Forms | Why |
|---|---|---|
| `memory_source` | all, range, positions | R indexing |
| `nifti_array_source` | all, range, positions | volumes by index list, features through a mask |
| `zarr_array_source` | all, range | chunked range reads; positions are decomposed into runs |
| `feature_mapped_source`, `validity_masked_source`, sparse entity and lifted row sources | all, range, positions | any selector resolves directly to operator rows, mask rows, or Matrix indexing |
| `source_view`, `counting_source`, `fault_source` | the child's | they forward what they receive |
| `row_bound_source`, `row_sharded_source` | the intersection of the children's | routing preserves the form; a composition can push down only what every child can |

Declarations are ordinary capability strings, so they serialize and pass the
existing source contract; `validate_array_source()` does not require them, and
an extension source that declares none has simply not certified any form.

## Consequences

- One law replaces four; the table of cases in
  `tests/testthat/test-selection-algebra.R` runs against every owner.
- `source_view` fingerprints changed (schema version 2, canonical form
  hashed). Plans built by earlier development builds do not match current
  selections. Semantic digests are unchanged.
- Raw sources reject repeated positions; `locate_source_rows()` does too.
- Collections accept negative positions.
- A view's assay descriptor, explain summary, and plan fingerprint no longer
  scale with a select-all or range axis.
- Extension sources are held to the law at the generic without changing their
  method signatures; `source_read(x, observations, features)` still accepts
  `NULL`, integer, and logical selectors. The algebra itself is internal: no
  constructor is exported.
