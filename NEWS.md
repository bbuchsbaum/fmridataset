# fmridataset 0.10.0 (Development)

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
* Added serializable balanced, imagewise, and featurewise block planners with
  explicit byte ceilings, chunk-aware block shapes, stale-plan detection, and
  bounded execution over frame views.
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
