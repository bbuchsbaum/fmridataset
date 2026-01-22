# Architecture

**Analysis Date:** 2026-01-22

## Pattern Overview

**Overall:** Modular S3 object system with pluggable storage backends and lazy evaluation

**Key Characteristics:**
- S3 class dispatch for polymorphic data access across multiple storage formats
- Storage backend interface (contract pattern) enabling format-agnostic operations
- Lazy evaluation via deferred array types (delarr, DelayedArray) for memory efficiency
- Hierarchical dataset types: single-subject datasets, multi-subject groups, and study-level aggregates
- Generic function contracts defined centrally, implementations spread across modular files
- Constructor pattern: `new_*()` (internal validation) → `*()` (user-facing)

## Layers

**S3 Generic Functions & Dispatch:**
- Purpose: Define contracts for dataset operations across all implementations
- Location: `R/all_generic.R`
- Contains: 30+ generic function definitions (get_data, get_mask, blocklens, data_chunks, etc.)
- Depends on: R base S3 system
- Used by: All dataset types, backend implementations, and data access methods

**Dataset Classes (S3):**
- Purpose: Unified representations of fMRI data from various sources
- Location: `R/fmri_dataset.R`, `R/dataset_constructors.R`, `R/fmri_group.R`
- Contains: Constructor functions and validation logic for:
  - `matrix_dataset` - In-memory matrix data
  - `fmri_mem_dataset` - NeuroVec-based volumetric data
  - `fmri_file_dataset` - File-backed data with backend support
  - `latent_dataset` - Latent space data from dimensionality reduction
  - `fmri_group` - Multi-subject grouped datasets
  - `fmri_study_dataset` - Composite study-level datasets
- Depends on: fmrihrf for sampling_frame, neuroim2 for spatial structures
- Used by: All downstream data access and processing methods

**Storage Backend Layer:**
- Purpose: Abstract stateful data access for different storage formats
- Location: `R/storage_backend.R` (interface), `R/*_backend.R` (implementations)
- Contract Methods: `backend_open()`, `backend_close()`, `backend_get_dims()`, `backend_get_mask()`, `backend_get_data()`, `backend_get_metadata()`
- Implementations:
  - `R/matrix_backend.R` - In-memory matrix storage
  - `R/nifti_backend.R` - NIfTI file-based access with per-file caching
  - `R/h5_backend.R` - HDF5 file format
  - `R/zarr_backend.R` - Zarr cloud-native format
  - `R/study_backend.R` - Composite multi-subject backend
  - `R/latent_backend.R` - Latent space (dimensionality-reduced) data
- Depends on: Specific file format libraries (neuroim2, hdf5r, etc.)
- Used by: Dataset constructors for data acquisition, data access methods for retrieval

**Data Access Layer:**
- Purpose: Implement generic contracts for retrieving data from datasets
- Location: `R/data_access.R`
- Contains: Method implementations for get_data*, get_data_matrix*, get_mask*, blocklens* across all dataset types
- Depends on: Backends for file access, neuroim2 for NeuroVec operations
- Used by: User-facing functions, data processing pipelines

**Data Chunking & Iteration Layer:**
- Purpose: Support memory-efficient processing via chunked data access
- Location: `R/data_chunks.R`
- Contains:
  - `data_chunk()` - Represents a single chunk subset
  - `chunk_iter()` - Iterator protocol for sequential chunk access
  - `data_chunks()` implementations - Strategy-based chunking (voxel-wise or run-wise)
  - Helper functions: `exec_strategy()`, `collect_chunks()`, `arbitrary_chunks()`, `slicewise_chunks()`, `one_chunk()`
- Depends on: Iterator protocol, backend data access
- Used by: Parallel/sequential processing workflows (foreach integration)

**Time Series Container (S4):**
- Purpose: Lazy time series representation with metadata
- Location: `R/FmriSeries.R`, `R/fmri_series.R`, `R/fmri_series_metadata.R`
- Contains:
  - `fmri_series` S4 class with slots: data (lazy matrix), voxel_info, temporal_info, selection_info, dataset_info
  - Constructor: `new_fmri_series()`
  - Methods for conversion and metadata access
- Depends on: delarr/DelayedArray for lazy evaluation, data_access methods
- Used by: Downstream analysis that requires spatial/temporal metadata alongside data

**Data Type Conversion Layer:**
- Purpose: Support flexible conversion between dataset types
- Location: `R/conversions.R`, `R/as_delayed_array.R`
- Contains: Methods implementing `as.matrix_dataset()`, `as_delayed_array()` conversions
- Depends on: Target format constructors, data access methods
- Used by: API flexibility for user workflows

**Temporal Structure Management:**
- Purpose: Manage run/block information and temporal metadata
- Location: `R/sampling_frame_adapters.R`
- Contains: Bridge functions between fmridataset and fmrihrf sampling_frame conventions
- Depends on: fmrihrf::sampling_frame class
- Used by: Dataset constructors, metadata access methods

**Display & Reporting:**
- Purpose: User-facing formatted output and diagnostics
- Location: `R/print_methods.R`
- Contains: print.* methods for fmri_dataset, latent_dataset, chunkiter, data_chunk
- Depends on: Base print system
- Used by: Interactive R sessions for object inspection

**Configuration & Error Handling:**
- Purpose: Global settings and uniform error reporting
- Location: `R/config.R`, `R/errors.R`, `R/zzz.R`
- Contains:
  - `default_config()`, `read_fmri_config()` - Configuration management
  - Custom error classes and `stop_fmridataset()` - Consistent error reporting
  - Package initialization in zzz.R
- Depends on: R base system, lifecycle for deprecations
- Used by: All other layers for settings and error generation

**Group-Level Operations:**
- Purpose: Multi-subject dataset operations and group analysis setup
- Location: `R/fmri_group.R`, `R/group_iter.R`, `R/group_map.R`, `R/group_verbs.R`, `R/group_stream.R`
- Contains: Group construction, iteration, mapping, and data frame operations
- Depends on: Dataset types, data access layer
- Used by: Multi-subject analysis workflows

## Data Flow

**Single-Subject File-Backed Read:**

1. User calls `fmri_dataset(backend = nifti_backend(...))`
2. `nifti_backend()` creates environment-based backend with source paths/objects
3. Constructor wraps backend in fmri_dataset list with metadata (TR, sampling_frame, etc.)
4. User calls `get_data_matrix(dataset)`
5. `get_data_matrix.fmri_file_dataset()` dispatches to `backend_get_data()`
6. Backend opens file (via `backend_open()`), reads specified rows/cols, returns matrix
7. Lazy evaluation via delarr not applied unless `as_delayed_array()` called

**Multi-Subject Group Processing:**

1. User creates data.frame with one row per subject, list column of datasets
2. User wraps with `fmri_group(subjects, id="subject_id", dataset_col="dataset")`
3. User calls `data_chunks(group_obj, nchunks=4)` or iterates via `group_iter()`
4. Operation iterates subjects, applies chunking per subject, returns iterator
5. User passes iterator to foreach for parallel processing
6. Each chunk carries voxel/row indices for reconstruction

**Lazy Array Access:**

1. User calls `as_delayed_array(dataset)` on any dataset type
2. Method dispatches based on dataset class
3. Constructs delarr object wrapping backend_get_data accessor
4. User operates on lazy array; materialization deferred until realization
5. Optional: pass to analysis functions expecting DelayedArray (e.g., from Bioconductor)

**State Management:**

- Datasets are primarily immutable lists with S3 class attributes
- Backends use environments (reference semantics) to maintain state (file handles, caches)
- `backend_open()` / `backend_close()` manage resource lifecycle
- Caching (via cachem, memoise) transparent to user code
- No global mutable state; all state contained in objects

## Key Abstractions

**Storage Backend:**
- Purpose: Abstract away format-specific details (NIfTI, HDF5, Zarr, matrix)
- Examples: `R/matrix_backend.R`, `R/nifti_backend.R`, `R/h5_backend.R`, `R/zarr_backend.R`
- Pattern: S3 class inheriting from "storage_backend", implements 6 required methods
- Benefit: New formats added without changing dataset API

**Dataset Type Hierarchy:**
- Purpose: Organize different data sources under common interface
- Base: All inherit from "fmri_dataset" S3 class
- Variants: matrix_dataset (simplest), fmri_mem_dataset (NeuroVec), fmri_file_dataset (backend-based), latent_dataset (latent space), fmri_group (grouped), fmri_study_dataset (composite)
- Benefit: Single generic function set works across all types via method dispatch

**Series Selector:**
- Purpose: Flexible specification of voxel subsets (by index, coordinate, ROI, sphere)
- Location: `R/series_selector.R`, `R/series_alias.R`
- Pattern: Generic `resolve_indices()` converts selector type to numeric indices
- Benefit: Compose selection without materializing intermediate data

**Sampling Frame (Temporal Structure):**
- Purpose: Encapsulate run lengths, TR, and temporal properties
- Source: fmrihrf::sampling_frame external class
- Usage: Stored in dataset for blocklens, n_runs, n_timepoints, block_ids access
- Benefit: Unified temporal metadata across all data sources

## Entry Points

**Package User API:**

- `fmri_dataset()` (generic constructor, dispatches based on inputs)
- `matrix_dataset(mat, TR, run_length)` - simplest in-memory construction
- `fmri_mem_dataset(scans, mask, TR, ...)` - NeuroVec-based construction
- `latent_dataset(source, TR, run_length, ...)` - latent space data
- `fmri_group(subjects, id, dataset_col)` - grouped datasets
- `nifti_backend()`, `h5_backend()`, `zarr_backend()` - backend constructors

**Core Operations:**

- `get_data(x)` - Extract raw data in native format
- `get_data_matrix(x)` - Extract as standard matrix (time × voxels)
- `get_mask(x)` - Extract voxel mask
- `data_chunks(x, nchunks, runwise)` - Create chunked iterator
- `as_delayed_array(x)` - Convert to lazy array
- `as.matrix_dataset(x)` - Convert to matrix_dataset

**Temporal Access:**

- `blocklens(x)`, `get_run_lengths(x)` - Run lengths in timepoints
- `n_runs(x)`, `n_timepoints(x)` - Temporal dimensions
- `get_TR(x)`, `get_total_duration(x)`, `get_run_duration(x)` - Timing info
- `blockids(x)`, `samples(x)` - Timepoint indexing vectors

**Group Operations:**

- `group_iter(group, f)` - Apply function to each subject
- `group_map(group, f, .id)` - Map with subject ID tracking
- `group_stream(group, f)` - Stream results from group processing

## Error Handling

**Strategy:** Centralized error system with custom condition classes and helpful messages

**Patterns:**

- Use `stop_fmridataset(error_config, message, ...)` throughout codebase
- Error configurations in `R/errors.R`: fmridataset_error_config, fmridataset_error_backend_io, etc.
- Backends validate inputs on construction before accepting data
- `validate_backend()` checks all required methods implemented
- Dataset constructors validate logical consistency (sum(run_length) == nrow(data), etc.)

## Cross-Cutting Concerns

**Logging:**
- No structured logging layer; use base print/message for diagnostics
- Some backends (nifti_backend) log cache hits/misses if option enabled

**Validation:**
- Assertthat assertions in constructors for preconditions
- S3 generic dispatch guarantees type-safe operations
- Backend contract enforced via method checks in `validate_backend()`

**Authentication:**
- Not applicable; all data is local or file-based
- Cloud formats (Zarr) rely on external storage authentication (AWS, GCS, etc.)

**Caching:**
- Per-backend caching via cachem (NIfTI metadata/masks)
- Memoization for expensive lookups (mask conversion to logical)
- Cache settings controllable via options: `fmridataset.cache_max_mb`, `fmridataset.cache_evict`, `fmridataset.cache_logging`

**Concurrency:**
- No built-in thread safety; R's fork-based parallelism safe for read-only operations
- Data chunks created for foreach %dopar% workflows
- Backends designed stateless where possible (matrix, zarr) or single-threaded (nifti)

---

*Architecture analysis: 2026-01-22*
