# fmridataset 0.10.0 (Development)

- Fixed a second set of review findings, on the contraction itself.
  `as_delarr()` bounds each pull rather than the whole array, so a source
  larger than the ceiling can be wrapped and read in chunks, which is what a
  finite budget is for; a nonsensical ceiling is still refused at the wrap.
  FDS v1 manifests written before the `id_policy` and typed-metadata fields
  are read again: the schema identity never changed, so an absent `id_policy`
  is read as the `require` policy (a persisted ID is supplied and durable) and
  plain-list container metadata is typed through `unaligned_record()`, which
  still rejects runtime state. `source_realization_cost()` is now a generic,
  and the wrapper sources whose reads materialize more than they return
  (`feature_mapped_source`, `validity_masked_source`) charge those
  intermediates, so a budget can no longer approve a one-column read that
  then allocates the contributing columns and the product temporaries.
  `filter_entities()` restricts entity-addressed relations alongside the
  registry, so a study carrying a per-entity validity mask or a sparse
  relation on the filtered entity can be filtered at all, and it filters
  typed tables stored on a frame by the same entity key it applies to
  study-level tables, so no stale rows survive the filter.
- Fixed a set of review findings. `bind_observations()` now compares entity
  registries semantically (names, keys, scalar data, block components, and
  block values, by fingerprint first and by realized values only when
  fingerprints differ), so frames reopened from FDS bind with each other and
  with their in-memory originals. Feature-mapped, validity-masked, fault,
  and row-sharded sources compute their fingerprints once at construction, as
  ADR-009 promises, instead of re-hashing the operator or mask bank on every
  `plan_blocks()` and `execute_block_plan()` call. `validate_array_source()`
  resolves protocol methods from the caller's scope, so an extension class
  defined inside `local()` or a test block validates. Canonical encoding of
  character vectors is vectorized (byte-for-byte unchanged; the golden
  vectors still hold) and axis frames cache the digest of their IDs, which
  makes `explain()`, `assays()` on a view, and manifest digests of wide
  frames orders of magnitude faster. `explain()` reports `ids_durable` and a
  `NULL` semantic digest for a frame with ephemeral IDs instead of aborting,
  while `identity_descriptor(domain = "semantic")` still refuses and names
  the ephemeral axis. Runtime state (functions, environments) in axis,
  axis-block, and assay metadata is rejected at construction. Every selection
  over an empty axis now has the one form `all`, so plans over an empty frame
  and its `integer()` subset agree, and `Inf`, `-Inf`, or magnitudes past the
  integer range are rejected as selectors with structured reasons
  (`non_finite`, `out_of_bounds`) rather than a bare coercion error.
  FDS manifests written by earlier 0.10 development builds, before the
  `id_policy` and typed-metadata fields, are not readable by this build and
  must be rewritten from the source data; the schema version stays 1 because
  no build with the earlier layout was released.
- Added `source_error()`, an exported constructor for the stale, I/O, and
  contract conditions an array source may signal, so storage packages fail
  the way built-in sources fail. `validate_array_source()` now names the
  protocol methods a descriptor lacks instead of failing with a bare
  no-applicable-method error. A caller-supplied `revision` on
  `memory_source()` now replaces the per-object identity token, so equal-valued
  sources built independently under the same revision share a fingerprint, as
  ADR-009 describes. `feature_map_from_target()` and `map_features()` explain
  that a synthesis-only basis has no analysis operator instead of failing on
  the operator type.
- Replaced the package's independent selector mechanisms with one selection
  algebra (`inst/architecture/ADR-010-selection-algebra.md`). Frames, views,
  source views, `source_read()`, collections, axis and entity frames, and
  `filter_entities()` now normalize selectors through one law: character
  selectors are stable IDs that must exist, be unique, and keep request
  order; logical selectors must match the axis with no `NA`; numeric
  selectors must be whole, may reorder or be negative but not mixed, drop
  zero, and must be in bounds; an element appears at most once; and an empty
  selection is legal everywhere (a collection still cannot be empty, as a
  container rule with `reason = "empty_collection"`). Errors carry a
  structured `reason`. Consequently raw sources and `locate_source_rows()`
  now reject repeated positions, the law is enforced at the `source_read()`
  and `source_read_native()` generics for extension sources too, and
  collections accept negative positions. Selections are stored in a compact
  normalized form (`all`, `range`, or `positions`): `source_view()` no longer
  stores expanded index vectors, nested source views and nested frame views
  compose into one view over the root, Zarr reads take chunk runs from the
  form instead of re-deriving them, `explain()` reports the selection form
  under `selection`, and descriptor size, fingerprint cost, and plan
  fingerprint cost no longer scale with a select-all or range axis.
  `source_view` fingerprints changed (schema version 2); `fds_manifest_digest()`
  is unaffected. `source_capabilities()` now reports the selector forms a
  backend pushes down natively as `pushdown:all`, `pushdown:range`, and
  `pushdown:positions`; every built-in source declares its forms.
- Rewrote the vignettes for the frame API. The pre-frame vignettes were
  removed with the legacy surface; the four replacements are
  `vignette("fmridataset")` (frames, views, laziness, the temporal contract,
  ID policy, and binding), `vignette("feature-spaces")` (volume, parcel,
  basis, and composite spaces, spatial identity, and feature maps),
  `vignette("persistence-and-import")` (FDS manifests, identity domains,
  fingerprints versus content hashes, the HDF5 round trip, and
  `read_bids_bold()`), and `vignette("extending-sources")` (implementing and
  validating an array source). All four run on small synthetic data and guard
  the `fmristore`, `bidser`, and `fmrihrf` examples on those packages being
  installed.
- Made source fingerprint and content-hash policy explicit
  (`inst/architecture/ADR-009-source-fingerprints-and-content-hashes.md`).
  `source_fingerprint()` is now a cheap revision fingerprint of the descriptor
  and its physical revision evidence, computed once at construction and cached,
  never of array values: `memory_source()` no longer hashes its payload and
  derives its fingerprint from shape, dtype, chunks, a per-object identity
  token, and an optional `revision`, so equal-valued memory sources built
  independently have different fingerprints by design; `identity = "content"`
  is the explicit opt-in. Sparse entity, lifted, and view fingerprints are
  cached so repeated calls are O(1), and canonical encoding of numeric vectors
  is vectorized. Added the exported extension generic `content_hash()` (with
  `content_hash_contract()`), an explicit O(n) SHA-256 of realized values
  streamed in bounded, chunk-aligned blocks that agrees across memory copies,
  storage dtypes, views, and row-bound compositions and is accepted as a
  content receipt by `identity_descriptor()`. Every file-backed source now
  raises `fmridataset_error_source_stale` (with `source`, `expected`, and
  `actual` fields) when its files or store changed after construction; NIfTI
  previously raised `fmridataset_error_backend_io`, which is now reserved for
  genuine I/O failures. `plan_blocks()` no longer rejects frames with an empty
  axis while estimating the per-value cost.
- Axis blocks are now two-dimensional: rows are the owning axis elements and
  columns are named components (`inst/architecture/ADR-008-axis-block-dimensionality.md`).
  `axis_block()` rejects vectors and arrays with more than two dimensions with
  a structured alignment error carrying `shape` and `dims`; axis, entity, and
  manifest validation name the offending block. FDS manifests no longer emit
  synthetic `dimension:` axis labels, and block arrays declaring trailing axes
  are rejected. This fixes `bind_observations()` silently flattening
  higher-dimensional blocks. Feature blocks of every bound operand must now
  agree with the first frame's components and values.
- Stable keys, scalar columns, unique names, one-string fields, runtime-state
  guards, block alignment, and synchronized subsetting are validated once, in
  shared internal helpers, across axes, entities, event and auxiliary tables,
  relations, and FDS manifests. Each domain keeps its existing error class and
  wording.
- Realization budgets now distinguish storage dtype and bytes from the R
  output dtype, retained output, temporary selection, conversion, or
  decompression buffers, and estimated peak working memory. The shared
  peak-cost contract is
  enforced by assay, chunk, block, spatial, and finite `delarr` collection
  paths, with counting sources reporting storage and realized traffic
  separately.
- The test suite now runs under testthat edition 3
  (`Config/testthat/edition: 3`), activating the previously inert snapshot
  tests for canonical serialization. `series()` now signals its deprecation
  through `lifecycle::deprecate_warn()`.
- `write_frame()` now returns the committed path normalized with forward
  slashes on every platform, and zarr `file://` sources are resolved to native
  filesystem paths before opening, fixing Windows-only failures.
- Classified the namespace into user, extension, and developer-only audiences
  (`inst/architecture/API-AUDIENCES.md`). `%||%` is no longer exported;
  counting and fault sources remain only as documented conformance tools for
  companion packages.
- Added one zero-I/O canonical frame schema (`frame_schema()`) for collection
  compatibility, observation binding, bounded explanation, FDS validation, and
  downstream protocol checks, with path-specific structured mismatch
  diagnostics.
- Made `bind_observations()` lossless and policy-driven. Compatible assay,
  axis, feature, entity, relation, and validity annotations are retained;
  frame metadata must match or be explicitly conflict-free merged; active
  assay differences require an explicit result; keyed typed tables union with
  conflict detection; and every bind creates a provenance node over all input
  graphs. Nested row-bound sources flatten canonically and empty views bind
  without forcing numerical reads.
- Made container metadata, typed tables, aligned values, and lineage
  mechanically distinct. Frame, collection, study, and FDS constructors now
  require `unaligned_record` metadata semantics and `provenance_graph` lineage;
  `as_provenance_graph()` converts a list of provenance records into a graph.
  Added `auxiliary_table()` for keyed files, contrasts, transforms, and other
  relational tables, and reject axis-length vectors, result diagnostics,
  arrays, and raw data frames hidden in generic metadata.
- Defined typed semantic, schema, space, source, provenance, and optional
  content identities under an explicit R-only canonicalization v1 contract.
  The package-owned tagged binary encoder now publishes exact golden bytes and
  SHA-256 vectors for numeric, Unicode, factor, dimension, sparse, and nested
  values; `stringi` is the sole added hard dependency for platform-independent
  UTF-8 NFC normalization.
  Added `same_space()` for exact spatial identity; the older compatibility
  names remain exact-identity aliases and never infer alignment from shape.
- Made axis identity policy explicit. Durable IDs are now supplied or derived
  deterministically from declared keys; UUID-backed IDs require an explicit
  `ephemeral` policy, are visibly marked, and are rejected by FDS persistence
  and semantic certification.
- Made `explain()` bounded for large axes: it now reports counts, source
  contracts, realization estimates, semantic/schema digests, and sampled IDs
  without numerical reads. Complete IDs require `ids = "complete"`.
- Frame views now expose assay descriptors for their visible rectangle:
  sources, shapes, and axis digests remain synchronized through reordered,
  composed, ID-selected, and empty views without reading numerical data.
- Added the canonical `as_fmri_frame()` coercion generic so companion packages
  can provide explicit legacy adapters without owning a competing frame type.
- Added `read_bids_bold()` as a narrow one-subject fMRIPrep on-ramp to a lazy
  `fmri_frame`, with deterministic relative-path volume IDs, exact selectors,
  explicit spatial ambiguity failures, common run-mask intersection, keyed
  events, and non-mutating discovery that reads BOLD headers and masks but no
  BOLD values.

## Architecture

* Began the 1.0 migration around a canonical observation-by-feature
  `fmri_frame`, with spatially typed features and serializable array sources.
* Recorded package ownership and compatibility policy in
  `inst/architecture/ADR-001-canonical-data-model.md`.
* Added `write_frame()` and `open_frame()` as semantic entry points for
  atomic, manifest-backed HDF5 persistence supplied by `fmristore`; reopened
  assays remain reconstructible lazy sources.
* Certified the first complete frame-native analysis path: metadata-only
  filtering, stimulus-block design compilation, bounded variance-aware group
  fitting, spatial-map reconstruction, and exact memory/HDF5 round trips.
* Added executable `ArraySource` contract validation for supported dtypes,
  bounded chunk grids, capabilities, stable fingerprints, and freedom from
  unserializable runtime state.
* Array sources now become reconstructible `delarr` provider seeds; serialized
  plans retain descriptors and selectors rather than pull closures or handles.
* Added a serializable NIfTI source with per-file volume pushdown, packed-mask
  feature selection, stale-file detection, native-volume reads, and direct
  `volume_space` recovery.
* Added manifest-backed `row_sharded_source()` descriptors with stable shard
  IDs, inspectable global-to-local row routing, exact touched-shard pushdown,
  immutable shard append, and a compatible `row_bound_source()` constructor.
* Added an experimental serializable `zarr_array_source()` for two-dimensional
  observation-by-feature stores, including explicit physical-axis order,
  metadata freshness checks, consecutive-range pushdown, optional runtime
  discovery, deterministic handle cleanup, and direct `delarr` compatibility.
* Added canonical `entity_frame` and `entity_registry` contracts with stable
  primary keys, scalar metadata, aligned multivariate blocks, synchronized
  subsetting, frame/view accessors, and source-free FDS entity-block arrays.
* Added validated `key_relation`, `sparse_relation`, and `relation_registry`
  contracts with explicit observation, feature, and entity domains,
  referential-integrity checks, view restriction, row-bind merging, and FDS
  persistence.

* Added assay-free `hierarchy_index()` derivation for explicit root-to-leaf
  containment paths, with entity-order-stable grouping codes, crossed-relation
  exclusion, ambiguity checks, missing-ancestry propagation, and lazy-view
  invariance.

* `observations(..., resolve = TRUE)` now exposes namespaced scalar annotations
  from every entity reachable through validated key relations, while
  `obs_blocks(..., resolve = TRUE)` provides lazy observation-aligned views of
  entity blocks without duplicating their stored rows. `filter_obs()` resolves
  entity annotations by default and still performs no assay reads.

* Added `fmri_collection` for named, semantically equivalent frames that must
  retain separate feature spaces, including participant-native data. Collection
  validation compares assay, axis, block, entity, and relation schemas without
  inferring spatial equality from dimensions, and inspection remains zero-read.

* Added `fmri_study`, canonical source-to-target `frame_link` descriptors,
  keyed `event_table` objects, shared entity contextualization, and
  self-contained lazy filtered studies. Entity filters propagate through
  frames and native-space collections and compact visible entities, linked
  axis maps, and typed table rows without reading assay data. Feature operators
  are first-class link fields; `compose_frame_links()`,
  `upgrade_frame_link()`, and `upgrade_fds_study_manifest()` make direction,
  composition, and provisional-schema migration explicit.
* Added serializable balanced, imagewise, and featurewise block planners with
  explicit byte ceilings, chunk-aware block shapes, stale-plan detection, and
  bounded execution over frame views.
* Added explicit matrix-versus-spatial execution dispatch. Complete feature
  domains use native source reads when available; restricted domains safely
  reconstruct through their feature space, with bounded streaming helpers.
* Froze the backend-neutral FDS logical manifest at version 1, including a
  named-axis array registry, source-free assay and block declarations, strict
  validation, semantic digests, physical binding, and frame reconstruction.
* Added `surface_space` with stable full-mesh vertex and hemisphere identity,
  packed active/medial-wall support, content-addressed topology and geometry,
  induced sparse adjacency, surface-map reconstruction, restriction, spatial
  compatibility, and source-free FDS persistence. Surface identity now follows
  `neurosurf`'s surface-to-world transform convention, with an explicit adapter
  to and native reconstruction path for `neurosurf::SurfaceGeometry`.
* Added parent-linked `parcel_space` with sparse membership, explicit mean/sum
  aggregation and reconstruction operators, induced parcel adjacency, stable
  atlas-namespaced feature IDs, restriction, and FDS persistence. The optional
  `neuroatlas` adapter delegates atlas metadata and atlas-specific surface label
  coding to `neuroatlas::as_parcel_data()` and `neuroatlas::get_roi()`.
* Added parent-linked `basis_space` with stable component identities, explicit
  analysis and synthesis operators, exact SVD-based least-squares projection
  for non-orthonormal dictionaries, restriction, reconstruction, provenance,
  backend-neutral in-memory identity, and FDS/HDF5 persistence. An optional
  `fmrilatent` adapter treats spatial loadings as the synthesis dictionary while
  leaving model fitting, temporal scores, handles, and offsets in `fmrilatent`.
* Added ordered heterogeneous `composite_space` domains for mixed surface,
  volume, parcel, and representational parts. Part-qualified feature IDs,
  explicit routing, arbitrary-order restriction, block-diagonal adjacency,
  named native reconstruction, and FDS/HDF5 persistence support
  grayordinate-like data without duplicating child-space geometry classes.
* Added serializable `feature_map` descriptors with exact source and target
  space identity, lazy target-by-source assay transformation, explicit squared
  weight propagation for independent variances, canonical parcel and basis
  maps, typed study-link validation, and content-addressed acyclic derivation
  provenance.
* Added bit-packed, deduplicated `mask_bank` storage and typed
  `entity_feature_validity` relations. Validity follows feature views, resolves
  lazily to observations, reports policy-free coverage, persists through FDS,
  and can mask selected assays with `NA` without conflating absent coverage
  with numerical zero.
* The historical `v0.9.0` tag is preserved; development continues from the
  current main line without retagging it.

## Bug fixes

* `block_apply()` now returns an empty list for a frame or view with no
  features, instead of failing with `wrong sign in 'by' argument`. An empty
  selection is a supported frame state, and the rest of the frame API already
  honoured it.
* `feature_ids()` on a `volume_space` with empty support now returns
  `character(0)` rather than the single string `"voxel-"`, so feature IDs stay
  aligned with `n_features()`. This desynchronisation made `explain()` and
  `fds_frame_manifest()` fail on any zero-feature view of a volume space.

## Dependencies

* Declared `Remotes` entries for `fmrilatent` and `fmristore`, and removed the
  archived `pryr` from `Suggests`. Without these, `pak` could not solve the
  dependency graph, so every CI workflow failed during dependency setup before
  reaching `R CMD check`.
* Constrained `delarr (>= 0.1.0.9000)` and `fmristore (>= 0.1.0.9000)`, the
  versions that first provide `delarr_provider_pull()` and `write_frame_h5()`.
  Older builds previously failed at namespace load or mid-test rather than at
  dependency resolution.
* Dropped the Bioconductor dependency surface. `DelayedArray` and
  `DelayedMatrixStats` are no longer suggested, and CI no longer installs
  `BiocManager`, `Rarr`, `rhdf5`, `DelayedArray`, or `S4Arrays`. Lazy array
  support is built on `delarr`, which this project owns.

## Breaking changes

* Removed the pre-frame dataset architecture. `fmri_dataset()`,
  `matrix_dataset()`, `fmri_mem_dataset()`, `fmri_file_dataset()`,
  `fmri_h5_dataset()`, `fmri_zarr_dataset()`, `fmri_study_dataset()`,
  `latent_dataset()`, `bids_h5_dataset()`, `compress_bids_study()`, the
  storage-backend protocol and registry (`storage_backend`, `backend_*()`,
  `register_backend()`), the sampling-frame accessors (`get_TR()`,
  `blocklens()`, `blockids()`, `n_runs()`, `n_timepoints()`, ...),
  `data_chunks()` and its execution strategies, `fmri_series()` and the
  selector API, `fmri_group()` and the group verbs, `read_fmri_config()`, and
  the vignette data generators are gone. `fmri_frame()` is the only data
  container; `temporal_schema()` and `as_sampling_frame()` replace the
  sampling-frame accessors, `collect_assay()`, `plan_blocks()`, and
  `as_delarr()` replace chunk iteration, and `fmri_collection()` and
  `fmri_study()` replace the study dataset and group. The last commit carrying
  the old surface is `3ae565e`; applications that still need it should pin
  that revision while they migrate. `fmri_frame` objects no longer inherit
  from `fmri_dataset`. Serialized 0.x objects are not migrated by this
  package: load them with the pinned revision, build an `fmri_frame` from the
  matrix and metadata, and persist it with `write_frame()`.
* `as_delarr()` now dispatches on `x` rather than `backend`, and is defined for
  array sources only.
* `fmrihrf` moved from Imports to Suggests. Only `as_sampling_frame()` needs
  it, and that function now fails with a structured error when it is absent.
* Retired the `DelayedArray` bridge. `as_delayed_array()` and its methods, the
  `StorageBackendSeed` and `StudyBackendSeed` classes, and
  `register_delayed_array_support()` are removed. `as_delarr()` provides the
  same lazy interface over the same backends (`matrix_backend`,
  `nifti_backend`, `study_backend`, and a default method) and is the supported
  replacement.
* `fmri_series()` no longer accepts `output = "DelayedMatrix"`; `output` is now
  `"fmri_series"` only. The returned object already carries a `delarr` lazy
  matrix payload, which `as_delarr()` exposes directly. Note that `delarr` is a
  hard dependency, so the previous `DelayedArray` fallback path was unreachable
  in any installable configuration.

# fmridataset 0.9.0

## New features

* Added `dummy_mode` parameter to `fmri_dataset()` and `nifti_backend()` (#3)
  - Allows creation of datasets with non-existent file paths for testing
  - Returns placeholder data (zeros) and standard dimensions
  - Useful for testing dependent packages without requiring actual data files
  - Enable with `dummy_mode = TRUE` in `fmri_dataset()` constructor
* Replaced the DelayedArray dependency with the lightweight `delarr` lazy
  matrix adapter
  - `fmri_series()` and study helpers now return `delarr` objects by default
  - Added `as_delarr()` generics for all storage backends and study adapters
  - Retained optional `as_delayed_array()` paths for explicit DelayedMatrix output

# fmridataset 0.8.9 (Hotfix)

## Critical fixes

* Added bounded memory cache to prevent unbounded memory growth (#1)
  - Memoization now uses `cachem` with configurable size limit (default 512MB)
  - Added `fmri_clear_cache()` function to manually clear cache
  - Cache size configurable via `options(fmridataset.cache_max_mb = 1024)`

* Added memory warnings and mitigation for study_backend (#2)
  - Warning when operations will load >1GB into memory
  - Automatic chunking for operations that would load >2GB
  - Recommends using `data_chunks()` for large datasets

# fmridataset 0.1.0

## New features

* Added comprehensive CI/CD pipeline with GitHub Actions
* Added test coverage reporting with codecov
* Added code style checking and automatic formatting
* Added issue and PR templates for better project management
* Implemented `as_tibble.fmri_study_dataset` with metadata optimization
* Added integration and performance tests for `fmri_study_dataset` workflow

## Bug fixes

* Fixed chunking edge case when `nchunks > number of voxels`
* Updated deprecated `with_mock()` calls to `with_mocked_bindings()`
* Fixed dimensional consistency issues in storage backends
* Resolved all test failures from package refactoring

## Documentation

* Added comprehensive README with badges and examples
* Improved package architecture documentation
* Added codecov configuration for coverage reporting
* New vignette "From Single-Subject to Study-Level Analysis" with performance guidelines and architectural diagram

## Internal changes

* Refactored monolithic codebase into modular architecture
* Improved test organization and coverage
* Enhanced error handling and validation
* Modernized CI/CD workflows and tooling 
