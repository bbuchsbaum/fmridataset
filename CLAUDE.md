# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Build and Check
```bash
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
```bash
# Run custom test suite
Rscript tests/run_tests.R

# Run testthat tests
devtools::test()
testthat::test_dir("tests/testthat")

# Run a single test file
testthat::test_file("tests/testthat/test-sampling-frame.R")
```

### Documentation
```bash
# Generate package documentation
devtools::document()
roxygen2::roxygenize()

# Build pkgdown website
pkgdown::build_site()
```

## High-Level Architecture

### Core Classes

1. **`fmri_dataset`** - Central S3 class for unified fMRI data representation
   - Supports multiple data sources: file paths, pre-loaded objects, matrices, BIDS datasets
   - Lazy loading and chunked iteration capabilities
   - Constructor: `fmri_dataset_create()` with various helper functions

2. **`sampling_frame`** - Temporal structure representation
   - Encapsulates TR, run lengths, and temporal properties
   - Bridges fmrireg and fmridataset conventions
   - Constructor: `sampling_frame()`

3. **`bids_facade`** - BIDS integration (Phases 1-3)
   - Basic BIDS project wrapping
   - Discovery and quality assessment features
   - Simple caching and parallel processing

### Key Design Patterns

1. **S3 Object System**
   - Generic functions in `R/aaa_generics.R`
   - Method dispatch pattern throughout
   - Constructor pattern: `new_*()` (internal) → `*()` (user-facing)

2. **Fluent Interface**
   - BIDS queries use method chaining: `bids_query() %>% subject("01") %>% task("rest")`

3. **Transformation System**
   - Modular preprocessing pipeline in `R/transformations.R`
   - Composable transformations via `transformation_pipeline()`
   - Backwards compatible with legacy `apply_preprocessing`

### File Organization

- `R/aaa_generics.R` - S3 generic function definitions (loaded first)
- `R/fmri_dataset_*.R` - Core dataset functionality split by concern
- `R/bids_facade_phase*.R` - BIDS implementation phases (1-3)
- `R/sampling_frame.R` - Temporal structure handling
- `R/utils.R` - Helper functions
- `tests/testthat/test-*.R` - Comprehensive test coverage

### Testing Strategy

The package uses extensive testthat testing covering:
- Constructor validation
- Data access patterns
- Chunking and iteration
- BIDS integration phases
- Edge cases and error handling

Note: Some tests may fail as placeholders for future implementation - this is intentional and documented in test files.

### Integration Points

- **neuroim2**: Optional dependency for NeuroVec objects
- **bidser**: Optional BIDS backend
- **iterators**: Core dependency for chunked data access
- **tibble**: Data frame representation

The architecture emphasizes loose coupling, allowing components to evolve independently while maintaining stable interfaces through S3 generics.