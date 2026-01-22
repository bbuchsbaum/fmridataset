# Codebase Structure

**Analysis Date:** 2026-01-22

## Directory Layout

```
fmridataset/
├── R/                              # Package implementation (45 files)
│   ├── all_generic.R              # S3 generic function declarations (loaded first alphabetically)
│   ├── FmriSeries.R               # S4 class for lazy time series
│   ├── fmri_dataset.R             # Main entry point documentation
│   ├── dataset_constructors.R     # Dataset creation functions
│   ├── dataset_methods.R          # Additional dataset methods
│   ├── data_access.R              # Data retrieval implementations
│   ├── data_chunks.R              # Chunking and iteration logic
│   ├── conversions.R              # Type conversion methods
│   ├── print_methods.R            # Display/formatting methods
│   ├── storage_backend.R          # Backend interface contract
│   ├── matrix_backend.R           # In-memory matrix storage backend
│   ├── nifti_backend.R            # NIfTI file-based backend
│   ├── h5_backend.R               # HDF5 file-based backend
│   ├── zarr_backend.R             # Zarr cloud-native backend
│   ├── zarr_dataset_constructor.R # Zarr dataset-specific constructors
│   ├── study_backend.R            # Multi-subject composite backend
│   ├── study_backend_seed_s3.R    # Lazy evaluation for study backend
│   ├── latent_backend.R           # Latent space (reduced) data backend
│   ├── latent_dataset.R           # Latent space dataset interface
│   ├── fmri_group.R               # Multi-subject group datasets
│   ├── group_iter.R               # Group iteration functions
│   ├── group_map.R                # Group mapping functions
│   ├── group_stream.R             # Group streaming functions
│   ├── group_verbs.R              # Group data frame operations
│   ├── fmri_series.R              # Additional fmri_series methods
│   ├── fmri_series_metadata.R     # Metadata handling for series
│   ├── fmri_series_resolvers.R    # Series resolution logic
│   ├── sampling_frame_adapters.R  # Temporal structure handling
│   ├── series_selector.R          # Voxel selection interface
│   ├── series_alias.R             # Series aliasing utilities
│   ├── as_delayed_array.R         # DelayedArray conversion
│   ├── as_delarr.R                # Delarr conversion
│   ├── config.R                   # Configuration management
│   ├── errors.R                   # Custom error classes
│   ├── utils.R                    # General utilities
│   ├── fmri_dataset_legacy.R      # Backward compatibility support
│   ├── fmri_study_dataset.R       # Study-level dataset class
│   ├── study_dataset_access.R     # Study dataset data access
│   ├── golden_data_generation.R   # Golden reference data creation
│   ├── vignette_helpers.R         # Helper functions for vignettes
│   ├── mask_standards.R           # Standard mask definitions
│   ├── backend_registry.R         # Backend lookup/registration
│   ├── zzz.R                      # Package initialization
│   └── as_delayed_array_dataset.R # Dataset-specific conversions
│
├── tests/testthat/                # Comprehensive test suite (50+ test files)
│   ├── test-*.R                   # Individual test files by component
│   ├── test_backend_*.R           # Backend-specific tests
│   ├── test_data_chunks*.R        # Chunking tests
│   ├── test_fmri_series*.R        # FmriSeries class tests
│   ├── test_integration.R         # Cross-component integration
│   ├── test_golden*.R             # Golden data and snapshot tests
│   ├── helper-*.R                 # Test helpers and fixtures
│   ├── golden/                    # Golden reference data
│   ├── _snaps/                    # Snapshot test expectations
│   └── _testthat.yml              # Test configuration
│
├── man/                           # Auto-generated R documentation (148 files)
│   └── *.Rd                       # Function references from roxygen
│
├── vignettes/                     # Long-form user documentation
│   ├── *.Rmd                      # Markdown-based vignettes
│   └── articles/                  # Additional articles
│
├── data-raw/                      # Raw data for package examples
│   └── *.R                        # Scripts to generate package data
│
├── inst/                          # Installed package resources
│   └── *.txt, *.Rds               # Example data, reference files
│
├── examples/                      # Standalone example scripts
│   └── *.R                        # Usage examples
│
├── docs/                          # pkgdown website source
│   └── *.md, *.yml                # Site configuration
│
├── .github/                       # GitHub Actions CI/CD
│   └── workflows/                 # Automated testing pipelines
│
├── DESCRIPTION                    # Package metadata and dependencies
├── NAMESPACE                      # Exported functions and imports
├── README.md                      # Project overview
├── CLAUDE.md                      # Development guidance
├── CONTRIBUTING.md                # Contribution guidelines
├── SECURITY.md                    # Security policy
├── NEWS.md                        # Release notes/changelog
└── .planning/codebase/            # Documentation for GSD orchestrator
    ├── ARCHITECTURE.md
    └── STRUCTURE.md
```

## Directory Purposes

**R/ (45 files):**
- Purpose: All package implementation code
- Contains: S3 generics, dataset classes, backends, data access, utilities
- Key principle: One major abstraction per file, alphabetical loading for generic precedence
- Files loaded first: all_generic.R, then others alphabetically (generic definitions must come before implementations)

**tests/testthat/ (50+ test files):**
- Purpose: Comprehensive unit and integration testing
- Contains: Test cases for each backend, data access method, class constructor
- Key files:
  - `test-*_backend.R` - Backend-specific test suites
  - `test_integration.R` - Cross-component validation
  - `helper-*.R` - Reusable test fixtures and data generators
  - `golden/` - Reference data for golden tests
  - `_snaps/` - Snapshot test expectations (auto-generated)
- Coverage: All public APIs, edge cases, error conditions

**man/ (auto-generated):**
- Purpose: Function reference documentation
- Generated from: roxygen2 comments in R/*.R files
- Do NOT edit directly: regenerate via `devtools::document()`
- Convention: One .Rd file per exported function

**vignettes/:**
- Purpose: Long-form tutorials and guides
- Format: R Markdown (.Rmd)
- Built to: HTML via knitr during `devtools::build_site()`
- Examples: Workflow guides, backend explanations, group operations

**data-raw/:**
- Purpose: Source scripts for example datasets
- Convention: R scripts that create and save package data
- Output: .Rds or .RData files saved to inst/

**inst/:**
- Purpose: Non-R resources installed with package
- Contains: Example data, reference files, templates
- Access in code: `system.file("filename", package="fmridataset")`

## Key File Locations

**Entry Points & Main Classes:**
- `R/all_generic.R` - S3 generic function definitions (30+ contracts)
- `R/fmri_dataset.R` - Module organization documentation
- `R/dataset_constructors.R` - fmri_dataset(), matrix_dataset(), fmri_mem_dataset()
- `R/fmri_group.R` - fmri_group() for multi-subject datasets
- `R/FmriSeries.R` - S4 class for lazy time series with metadata

**Storage Backend Interface:**
- `R/storage_backend.R` - Contract definition (backend_open, backend_close, backend_get_dims, backend_get_mask, backend_get_data, backend_get_metadata)
- `R/matrix_backend.R` - In-memory matrix backend
- `R/nifti_backend.R` - NIfTI file backend with caching
- `R/h5_backend.R` - HDF5 file backend
- `R/zarr_backend.R` - Zarr array backend (cloud-native)
- `R/study_backend.R` - Composite multi-subject backend
- `R/latent_backend.R` - Latent space data backend

**Data Access & Retrieval:**
- `R/data_access.R` - Methods: get_data.*, get_data_matrix.*, get_mask.*, blocklens.*
- `R/data_chunks.R` - Chunking strategies: data_chunk(), chunk_iter(), data_chunks()
- `R/conversions.R` - Type conversions: as.matrix_dataset()
- `R/as_delayed_array.R` - Lazy array conversions: as_delayed_array()

**Temporal & Metadata:**
- `R/sampling_frame_adapters.R` - Bridge to fmrihrf::sampling_frame
- `R/fmri_series_metadata.R` - Voxel and temporal metadata handling
- `R/series_selector.R` - Voxel selection: resolve_indices()

**Group Operations:**
- `R/fmri_group.R` - Group construction and validation
- `R/group_iter.R` - Iteration over subjects
- `R/group_map.R` - Mapping functions over subjects
- `R/group_stream.R` - Streaming results
- `R/group_verbs.R` - Data frame operations on groups

**Display & Utilities:**
- `R/print_methods.R` - print.fmri_dataset, print.latent_dataset, print.data_chunk
- `R/config.R` - Configuration, default settings
- `R/errors.R` - Custom error classes: stop_fmridataset()
- `R/utils.R` - General-purpose utilities

**Configuration:**
- `DESCRIPTION` - Package metadata, dependencies (R >= 4.3.0, imports: neuroim2, delarr, fmrihrf, Matrix, etc.)
- `NAMESPACE` - Exported functions and imported symbols
- `.lintr` - Code style settings
- `.Rbuildignore` - Files excluded from built package

## Naming Conventions

**Files:**
- Core modules: snake_case (e.g., `data_access.R`, `matrix_backend.R`)
- Test files: `test-` or `test_` prefix (e.g., `test_dataset.R`, `test-nifti_backend.R`)
- Helper files in tests: `helper-` prefix (e.g., `helper-golden.R`)
- Auto-generated: `zzz.R` for .onLoad, `aaa_*.R` for early loading

**Functions:**
- Public API: snake_case, verbs or nouns (e.g., `get_data_matrix()`, `data_chunks()`, `fmri_dataset()`)
- Internal helpers: snake_case with `_impl` suffix (e.g., `get_chunk_impl()`)
- Constructors: `new_*()` for internal validation-free, `*()` for public with validation
- Methods: `method.classname` pattern (e.g., `get_data.matrix_dataset()`)
- Generics: snake_case (e.g., `backend_open()`, `blocklens()`)

**Classes:**
- S3 classes: character string in class vector (e.g., "fmri_dataset", "storage_backend", "data_chunk")
- S4 classes: CamelCase (e.g., "FmriSeries", "NeuroVec" from neuroim2)
- Backend classes: append "_backend" (e.g., "matrix_backend", "nifti_backend", "study_backend")
- Dataset variants: "type_dataset" pattern (e.g., "matrix_dataset", "fmri_mem_dataset", "fmri_file_dataset")

**Variables:**
- Column names in data.frames: snake_case (e.g., `voxel_info`, `temporal_info`)
- Matrix dimensions: `n_`, `n_timepoints`, `n_voxels`, `n_runs`
- Indices: `ind` suffix (e.g., `voxel_ind`, `row_ind`)
- Counts: numeric (e.g., `chunk_num`)

## Where to Add New Code

**New Backend Implementation:**
- File: `R/newformat_backend.R`
- Must implement: All 6 storage_backend contract methods
- Register in: `R/backend_registry.R` for discovery
- Tests: `tests/testthat/test-newformat_backend.R`
- Example: Study above `R/h5_backend.R` or `R/zarr_backend.R` for reference

**New Dataset Type:**
- Constructor file: `R/newtype_dataset_constructors.R` or add to `R/dataset_constructors.R`
- Data access: Add methods to `R/data_access.R` (get_data.newtype_dataset, get_data_matrix.newtype_dataset, etc.)
- Tests: `tests/testthat/test-newtype_dataset.R` with full coverage of new methods
- Documentation: roxygen2 comments in constructor file

**New Generic Operation:**
- Add generic definition to `R/all_generic.R` with @export and full documentation
- Implement methods in appropriate modular file (e.g., data_access.R for retrieval, data_chunks.R for chunking)
- Example: `get_data()` generic + get_data.matrix_dataset(), get_data.fmri_mem_dataset() implementations

**Utility Functions:**
- Shared helpers: `R/utils.R`
- Backend-specific: Within corresponding backend file
- Series-specific: `R/series_selector.R` or `R/fmri_series_metadata.R`

**Tests:**
- Pattern: `tests/testthat/test-{feature}.R`
- Fixtures: Define or source from `tests/testthat/helper-*.R`
- Golden data: Add reference data to `tests/testthat/golden/` and use `helper-golden.R` functions
- Snapshots: Auto-created in `tests/testthat/_snaps/` on first run

## Special Directories

**tests/testthat/golden/:**
- Purpose: Reference data for golden/snapshot testing
- Format: .Rds files with pre-computed expected outputs
- Usage: Load via `load_golden_data()` from helper-golden.R
- Committed: Yes, enables reproducible test validation
- Generated: Scripts in `R/golden_data_generation.R`

**tests/testthat/_snaps/:**
- Purpose: Snapshot test expectations (output comparisons)
- Generated: Automatically by testthat::snapshot_* functions
- Committed: Yes, enables detecting output changes
- Update: `testthat::snapshot_accept()` after intentional changes

**data-raw/:**
- Purpose: Scripts to generate inst/ example data
- Usage: `source("data-raw/script.R")` creates file in inst/
- Committed: Yes, makes example data generation reproducible
- Do NOT commit .Rds output; regenerate via script

**docs/ (pkgdown):**
- Purpose: Website generation source files
- Contains: _pkgdown.yml (site config), reference ordering, vignette organization
- Generated: Static HTML in docs/ directory via `pkgdown::build_site()`
- Deployed: To GitHub Pages automatically

**vignettes/:**
- Purpose: Long-form documentation and tutorials
- Format: R Markdown (.Rmd) with executable R code blocks
- Built: To HTML during package build/check
- Access: Via `browseVignettes("fmridataset")` or pkgdown site

---

*Structure analysis: 2026-01-22*
