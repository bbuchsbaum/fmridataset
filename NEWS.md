# fmridataset 0.10.0 (Development)

## Architecture

* Began the 1.0 migration around a canonical observation-by-feature
  `fmri_frame`, with spatially typed features and serializable array sources.
* Recorded package ownership and compatibility policy in
  `inst/architecture/ADR-001-canonical-data-model.md`.
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
