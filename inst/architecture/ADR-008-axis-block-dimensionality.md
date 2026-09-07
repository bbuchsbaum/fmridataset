# ADR-008: Axis blocks are two-dimensional

## Status

Accepted for the 1.0 development line. Recorded before implementation of the
shared keyed-domain validators (`R/keyed-domain.R`).

## Context

An `axis_block` carries multivariate values aligned with one axis: observation
motion regressors, feature embeddings, per-entity scores. The constructor
accepted any array whose first dimension matched the axis, and the FDS writer
labelled every dimension beyond the second with a synthetic
`dimension:<key>:<i>` axis name. Nothing described what those trailing
dimensions meant, how they were keyed, or how they should be subset, compared,
or bound.

Two defects followed directly:

- `bind_observations()` row-bound block data with `rbind()`. On a
  three-dimensional array `rbind()` treats each operand as a vector, so two
  `2 x 2 x 3` blocks became one `2 x 12` matrix. The result silently desynced
  the second dimension from the block's component metadata.
- Persisted manifests declared axes that carried no identity. Two frames
  could agree on every declared label and still disagree on what the trailing
  positions represented.

The frame contract forbids anonymous semantic dimensions: every axis of every
array has explicit IDs, metadata, and alignment rules. A trailing block axis
without typed metadata is exactly that.

## Decision

Axis blocks are limited to two dimensions.

- Rows are the elements of the owning axis (observations, features, or the
  entities of one `entity_frame`).
- Columns are named components described by the block's `components` table,
  keyed by `.component_id`.

Higher-order structure is expressed with the representations the model
already types: as additional named components (flatten the trailing axis into
labelled columns), as several blocks (one per level of the trailing axis), or
as an assay when the structure is observation-by-feature.

The alternative, typed metadata for every trailing axis, was rejected. It
would add an axis-descriptor vocabulary, subsetting semantics, bind semantics,
and manifest fields for a case no current fixture or workflow needs, and it
would create a second way to express what components and assays already
express.

## Consequences

- `axis_block()` rejects data that is not two-dimensional with a structured
  `fmridataset_error_alignment` error carrying `shape` and `dims`. Vectors are
  rejected too; pass `matrix(x, ncol = 1)` for a single component. Array
  sources are two-dimensional by the source contract and pass unchanged.
- `axis_frame()`, `entity_frame()`, and `validate_entity_registry()`
  re-validate every block's shape and alignment and name the offending block
  (`block`) and, for entities, the owning entity (`entity`).
- FDS manifests declare exactly two axes for every block array: the owning
  axis and `component:<array key>`. Synthetic `dimension:` labels are no longer
  emitted, and `validate_fds_manifest()` and `validate_fds_study_manifest()`
  reject block arrays that declare more than two axes with an
  `fmridataset_error_schema` error naming the block.
- `bind_observations()` row-binds validated two-dimensional blocks. The shared
  helper checks every operand's shape and component count before calling
  `rbind()`, so flattening cannot recur.
- The only higher-dimensional fixture, a `7 x 2 x 3` tensor block in
  `test-fds-schema.R`, was deleted. It exercised the synthetic labels rather
  than any user workflow. It is replaced by tests that assert the two-axis
  declaration and the rejection of hand-declared trailing axes.
- The canonical frame schema keeps its `trailing_shape` field for blocks. It
  now always has length one (the component count); the field name is
  unchanged so that schema digests and comparisons are unaffected.
