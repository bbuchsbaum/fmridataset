# fmridataset 0.10.0 (Development)

- Defined typed semantic, schema, space, source, provenance, and optional
  content identities under an explicit R-only canonicalization v1 contract.
  Added `same_space()` for exact spatial identity; the older compatibility
  names remain exact-identity migration aliases and never infer alignment from
  shape.

- Added one zero-I/O canonical frame schema for collection compatibility,
  observation binding, bounded explanation, FDS validation, and downstream
  protocol checks, with path-specific structured mismatch diagnostics.

- Reduced the hard dependency surface to nine demonstrated runtime packages,
  added minimum versions and immutable commit pins for custom dependencies,
  and removed companion consumers from `Suggests` and `Remotes`. Companion
  integration remains available through separately certified runtime discovery.

- Classified the 1.0 namespace into user, extension, and developer-only
  audiences. Removed synthetic vignette helpers and `%||%` from the public
  surface; retained counting and fault sources only as documented conformance
  tools for companion packages.

- Made `explain()` bounded for large axes: it now reports counts, source
  contracts, realization estimates, semantic/schema digests, and sampled IDs
  without numerical reads. Complete IDs require `ids = "complete"`; counting
  and fault sources are documented as developer-only conformance tools.

- Froze the historical dataset/backend/sampling-frame architecture in
  `legacy-0x/` and removed it from the core namespace. The canonical package
  no longer imports `fmrihrf`; storage extensions now implement serializable
  `ArraySource` and FDS protocols instead of the open-handle backend registry.

- Detached canonical `fmri_frame` and `fmri_view` objects from legacy
  `fmri_dataset` dispatch. `upgrade_dataset()` now provides checked migration
  for provisional frames and self-contained 0.x matrix, series, and NeuroVec
  inputs while rejecting ambiguous backend datasets without numerical reads.

- Frame views now expose assay descriptors for their visible rectangle:
  sources, shapes, and axis digests remain synchronized through reordered,
  composed, ID-selected, and empty views without reading numerical data.

- Added the canonical `as_fmri_frame()` coercion generic so companion packages
  can provide explicit legacy adapters without owning a competing frame type.

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

* Added `fmri_study`, typed `frame_link` descriptors, keyed `event_table`
  objects, shared entity contextualization, and lazy `filter_entities()` study
  views. Entity filters propagate through frames and native-space collections,
  and restrict linked axis maps and event rows without reading assay data.
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
