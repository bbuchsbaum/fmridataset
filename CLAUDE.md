# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Common Development Commands

### Build and Check

``` bash
# Build package
R CMD build .
devtools::build()

# Check package (CRAN compliance)
R CMD check
devtools::check()

# Install package locally
R CMD INSTALL .
devtools::install()
```

### Testing

``` bash
# Run testthat tests
devtools::test()
testthat::test_dir("tests/testthat")

# Run a single test file
testthat::test_file("tests/testthat/test-sampling-frame.R")

# Run tests matching a pattern
devtools::test(filter = "backend")

# Run with coverage
covr::package_coverage()
```

### Documentation

``` bash
# Generate package documentation
devtools::document()
roxygen2::roxygenize()

# Build pkgdown website
pkgdown::build_site()
```

### Development Workflow

``` r
# Load all functions for interactive development
devtools::load_all()

# Check for common issues
devtools::check()

# Run specific checks
rcmdcheck::rcmdcheck(args = "--no-manual")
```

## High-Level Architecture

### Core Classes

1.  **`fmri_dataset`** - Central S3 class for unified fMRI data
    representation
    - Supports multiple data sources: file paths, pre-loaded objects,
      matrices, BIDS datasets
    - Lazy loading and chunked iteration capabilities
    - Main constructor:
      [`fmri_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_dataset.md)
      with backend-specific helpers
2.  **`FmriSeries`** - S4 class for lazy time series access
    - Built on DelayedArray/DelayedMatrix for memory efficiency
    - Contains both data and spatial/temporal metadata
    - Access via
      [`as_delayed_array()`](https://bbuchsbaum.github.io/fmridataset/reference/as_delayed_array.md),
      `as_tibble()` methods
3.  **`sampling_frame`** - Temporal structure representation
    - Encapsulates TR, run lengths, and temporal properties
    - Bridges fmrireg and fmridataset conventions
    - Constructor: `sampling_frame()`
4.  **Storage Backends** - Pluggable data access layer
    - `matrix_backend` - In-memory matrix storage
    - `nifti_backend` - NIfTI file access (optimized with caching)
    - `h5_backend` - HDF5 storage
    - `zarr_backend` - Zarr array format (cloud-native, chunked storage)
    - `study_backend` - Multi-subject study data

### Key Design Patterns

1.  **S3 Object System**
    - Generic functions in `R/all_generic.R` (loaded first
      alphabetically)
    - Method dispatch pattern throughout
    - Constructor pattern: `new_*()` (internal) → `*()` (user-facing)
    - Validation pattern: constructors validate inputs
2.  **Storage Backend Interface**
    - Contract defined in `storage_backend.R`
    - Required methods:
      [`backend_open()`](https://bbuchsbaum.github.io/fmridataset/reference/backend_open.md),
      [`backend_close()`](https://bbuchsbaum.github.io/fmridataset/reference/backend_close.md),
      [`backend_get_dims()`](https://bbuchsbaum.github.io/fmridataset/reference/backend_get_dims.md),
      [`backend_get_data()`](https://bbuchsbaum.github.io/fmridataset/reference/backend_get_data.md)
    - Backend validation via
      [`validate_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_backend.md)
    - Lazy loading pattern for efficient memory use
3.  **Chunking/Iterator Pattern**
    - [`data_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.md)
      for memory-efficient processing
    - Supports both voxel-wise and run-wise chunking strategies
    - Iterator protocol for sequential data access

### File Organization

**Core Generic Functions:** - `R/all_generic.R` - S3 generic function
definitions (loaded first)

**Data Structures:** - `R/FmriSeries.R` - S4 class for lazy time
series - `R/fmri_dataset.R` - Core dataset class -
`R/dataset_constructors.R` - Dataset creation functions -
`R/sampling_frame_adapters.R` - Temporal structure handling

**Storage Backends:** - `R/storage_backend.R` - Backend interface
definition - `R/matrix_backend.R` - In-memory matrix storage -
`R/nifti_backend.R` - NIfTI file backend (with caching) -
`R/h5_backend.R` - HDF5 storage backend - `R/zarr_backend.R` - Zarr
array backend (cloud-native) - `R/study_backend.R` - Multi-subject study
backend - `R/study_backend_seed.R` - Lazy evaluation for study backend

**Latent Space Interface:** - `R/latent_dataset.R` - Specialized
interface for latent space data

**Data Access & Processing:** - `R/data_access.R` - Data retrieval
methods - `R/data_chunks.R` - Chunking functionality -
`R/conversions.R` - Type conversion methods - `R/as_delayed_array.R` -
DelayedArray conversions

**Utilities:** - `R/config.R` - Configuration management -
`R/errors.R` - Custom error classes - `R/print_methods.R` - Display
methods for objects

### Testing Strategy

The package uses comprehensive testthat testing with 40+ test files:

**Test Organization:** - `test-*_backend.R` - Backend-specific tests -
`test-dataset_constructors.R` - Constructor validation -
`test-data_chunks*.R` - Chunking functionality -
`test-fmri_series_*.R` - FmriSeries class tests - `test-integration.R` -
Cross-component integration - `test-error_constructors.R` - Error
handling - `test-backward_compatibility.R` - Legacy API support

**Coverage Areas:** - All backend implementations - Data access patterns
and chunking strategies - Type conversions and metadata handling - Edge
cases and error conditions - Performance optimizations (e.g., NIfTI
caching)

### Integration Points

**Core Dependencies:** - **neuroim2**: NeuroVec objects and neuroimaging
data structures - **DelayedArray**: Lazy array operations for
FmriSeries - **Matrix**: Sparse matrix support - **iterators**: Chunked
data iteration

**Optional Dependencies:** - **bidser**: BIDS dataset integration -
**fmristore**: Advanced storage backends - **arrow**: Parquet file
support - **dplyr**: Data manipulation

The architecture emphasizes loose coupling through S3 generic functions,
allowing backends and components to evolve independently while
maintaining stable APIs.
