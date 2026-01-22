# Technology Stack

**Analysis Date:** 2026-01-22

## Languages

**Primary:**
- R 4.3.0+ - Core package language for all functionality

## Runtime

**Environment:**
- R 4.3.0 or higher (specified in `DESCRIPTION`)

**Package Manager:**
- R package system with Roxygen2 documentation generation
- Remotes specified for custom GitHub packages in `DESCRIPTION` Remotes field

## Frameworks

**Core:**
- S3 object system - Primary class system for all data structures
- S4 classes (FmriSeries) - For lazy time series access
- roxygen2 (version 7.3.2.9000) - Documentation generation

**Testing:**
- testthat (>= 3.0.0) - Unit testing framework

**Build/Dev:**
- devtools - Package development tools
- pkgdown - Website documentation generation
- knitr/rmarkdown - Vignette building

## Key Dependencies

**Critical (Imports):**
- assertthat - Input validation and assertions throughout codebase (`R/dataset_constructors.R`, `R/series_selector.R`)
- cachem - In-memory caching for metadata and masks, used in `R/nifti_backend.R`
- deflist - Default list implementation
- delarr - Custom DelayedArray integration wrapper
- fmrihrf - Temporal sampling frame functionality (`R/sampling_frame_adapters.R`), requires remote from `bbuchsbaum/fmrihrf`
- fs - File system operations (`R/dataset_constructors.R`)
- lifecycle - Deprecation warnings for API changes
- memoise - Function memoization for caching results
- Matrix - Sparse matrix support for latent space data
- methods - S4 class system for FmriSeries
- neuroim2 - Neuroimaging data structures (NeuroVec, NeuroVol), core integration point
- purrr - Functional programming utilities for data mapping
- tibble - Data frame enhancements for tabular data
- utils - Standard R utilities

**Optional Dependencies (Suggests):**
- DelayedArray - Lazy array operations for memory-efficient access, conditionally loaded in `R/zzz.R`
- hdf5r - HDF5 file reading (required when using h5_backend)
- Rarr - Zarr array format support (required for zarr_backend, `R/zarr_backend.R`)
- arrow - Parquet file format support
- dplyr - Data manipulation (optional for group operations in `R/group_map.R`, `R/group_verbs.R`)
- fmristore - H5 neuroimaging file format (remote from `bbuchsbaum/fmristore`), required for h5_backend
- bidser - BIDS dataset integration (remote from `bbuchsbaum/bidser`), enables BIDS constructor support
- foreach - Parallel iteration support
- mockery/mockr - Mocking framework for tests
- jsonlite - JSON configuration file parsing in `R/config.R`
- yaml - YAML configuration file parsing in `R/config.R`
- bench - Benchmarking utilities
- crayon - Colored terminal output
- rmarkdown - R Markdown support for documentation

**Remote Dependencies:**
- bbuchsbaum/delarr - Custom DelayedArray wrapper
- bbuchsbaum/fmrihrf - HRF and temporal sampling functionality
- bbuchsbaum/fmristore - H5 file format support
- bbuchsbaum/bidser - BIDS dataset support

## Configuration

**Environment:**
- Package options set in `R/zzz.R` via `.onLoad()`:
  - `fmridataset.cache_max_mb`: Main cache size (512MB default)
  - `fmridataset.cache_evict`: Cache eviction policy (LRU default)
  - `fmridataset.cache_logging`: Cache logging (disabled by default)
  - `fmridataset.study_cache_mb`: Study backend cache (1024MB default)
  - `fmridataset.block_size_mb`: Chunked operation block size (64MB default)
  - `fmridataset.cache_threshold`: Cache size threshold (10% default)

**Build:**
- `_pkgdown.yml` - pkgdown website configuration
- `codecov.yml` - Code coverage reporting configuration
- `.lintr` - lintr static analysis configuration
- `.Rbuildignore` - Files excluded from package build
- `.Rproj` - R project configuration

**Configuration File Support:**
- YAML format (`.yaml`, `.yml`) via `yaml` package - `read_fmri_config()` in `R/config.R`
- JSON format (`.json`) via `jsonlite` package - `read_fmri_config()` in `R/config.R`
- Legacy DCF format for backward compatibility - `read_dcf_config()` in `R/config.R`

## Platform Requirements

**Development:**
- R >= 4.3.0
- Roxygen2 for documentation generation
- devtools for development workflow
- System libraries for NIfTI support (via neuroim2)

**Production:**
- R >= 4.3.0
- Core dependencies as specified in DESCRIPTION Imports
- Optional dependencies installed as needed for specific backends

---

*Stack analysis: 2026-01-22*
