# External Integrations

**Analysis Date:** 2026-01-22

## APIs & External Services

**No HTTP-based external APIs:**
- This is a pure data handling library with no direct HTTP/REST API integrations
- No external API calls in the codebase

## Data Storage

**Supported Storage Formats:**

1. **NIfTI Format** (Neuroimaging Informatics Technology Initiative)
   - Backend: `nifti_backend` (`R/nifti_backend.R`)
   - Integration: via `neuroim2` package
   - Supports: File paths to NIfTI images, pre-loaded neuroim2 NeuroVec objects
   - Caching: cachem in-memory cache for metadata and masks
   - Modes: normal, bigvec, mmap, filebacked
   - Registration: Built-in, registered in `register_builtin_backends()`

2. **HDF5 Format** (Hierarchical Data Format 5)
   - Backend: `h5_backend` (`R/h5_backend.R`)
   - Integration: via `fmristore` package (conditional dependency)
   - Client: hdf5r package (required when using h5_backend)
   - Supports: H5 file paths, H5NeuroVec objects
   - Data path: Configurable via `data_dataset` parameter (default: "data")
   - Mask path: Configurable via `mask_dataset` parameter (default: "data/elements")
   - Notes: Requires separate `fmristore` package installation
   - Registration: Built-in, registered in `register_builtin_backends()`

3. **Zarr Format** (Cloud-native array format)
   - Backend: `zarr_backend` (`R/zarr_backend.R`)
   - Integration: via `Rarr` package (Bioconductor)
   - Supports: Local Zarr stores (directories/zip), remote S3/GCS/Azure URLs
   - Data key: Configurable (default: "data")
   - Mask key: Configurable (default: "mask"), optional
   - Chunking: Built-in chunk caching support (100 chunks default)
   - Features: Cloud-native, parallel read/write, progressive data access
   - Registration: Built-in, registered in `register_builtin_backends()`

4. **In-Memory Matrix Format**
   - Backend: `matrix_backend` (`R/matrix_backend.R`)
   - Supports: Native R matrices and Matrix sparse matrices
   - Stateless backend (no resource management needed)
   - Registration: Built-in, registered in `register_builtin_backends()`

5. **Latent Space Format**
   - Backend: `latent_backend` (`R/latent_backend.R`)
   - Integration: via `fmristore` LatentNeuroVec objects
   - Supports: Dimension-reduced data representations
   - Registration: Built-in, registered in `register_builtin_backends()`

6. **BIDS Format** (Brain Imaging Data Structure)
   - Integration: via `bidser` package (optional, remote: `bbuchsbaum/bidser`)
   - Constructor: `fmri_dataset()` accepts BIDS dataset paths
   - Support: Conditional on bidser package availability

**Study/Multi-Subject Format:**
   - Backend: `study_backend` (`R/study_backend.R`)
   - Supports: Multiple subject data across different storage backends
   - Features: Lazy evaluation of subject data, memory-efficient iteration
   - Registration: Built-in, registered in `register_builtin_backends()`

**Legacy Format:**
   - Support for pre-loaded `NeuroVec` objects from earlier versions
   - Via `fmri_mem_dataset()` constructor

## Caching & Performance

**Caching Infrastructure:**
- `cachem::cache_mem()` - In-memory cache for metadata
- `memoise` - Function-level memoization for repeated computations
- NIfTI backend cache: Metadata and mask caching via cachem
- Study backend cache: Configurable cache for subject data (default: 1024MB)
- Global options for cache tuning in `.onLoad()` (`R/zzz.R`)

## Metadata & Configuration

**Configuration File Support:**
- YAML format (`.yaml`, `.yml`) - via `yaml` package in `read_fmri_config()`
- JSON format (`.json`) - via `jsonlite` package in `read_fmri_config()`
- Legacy DCF format for backward compatibility in `read_dcf_config()`

**Metadata Handling:**
- NIfTI metadata: affine transformation, voxel dimensions, intent codes via neuroim2
- H5 metadata: Dataset-specific metadata extraction
- Zarr metadata: Stored as Zarr attributes
- Generic metadata interface: `backend_get_metadata()` in `storage_backend.R`

## Extensibility

**Backend Registry System** (`R/backend_registry.R`):
- Pluggable architecture allowing external packages to register custom backends
- Registration via `register_backend()` function
- Supports custom factory functions and validation functions
- All built-in backends registered in `register_builtin_backends()`
- Built-in backends: nifti, h5, matrix, latent, study, zarr

**Storage Backend Contract** (`R/storage_backend.R`):
Required methods all backends must implement:
- `backend_open()` - Acquire resources (e.g., file handles)
- `backend_close()` - Release resources
- `backend_get_dims()` - Return spatial (x,y,z) and time dimensions
- `backend_get_mask()` - Return logical mask of valid voxels
- `backend_get_data()` - Read data in timepoints × voxels orientation
- `backend_get_metadata()` - Return format-specific metadata

## CI/CD & Deployment

**GitHub Actions Workflows** (`.github/workflows/`):
- `R-CMD-check.yaml` - Standard R package checks across multiple OS/R versions
- `test-coverage.yaml` - Code coverage reporting via codecov
- `test-full-matrix.yaml` - Full matrix testing across configurations
- `lint.yaml` - Code linting
- `style.yaml` - Code style checks
- `pkgcheck.yaml` - pkgdown checks
- `pkgdown.yaml` - Website deployment
- `vignette-quality.yaml` - Vignette quality checks

**Dependency Management:**
- `.github/dependabot.yml` - Automated dependency updates

**Code Coverage:**
- `codecov.yml` - Codecov integration configuration

## Lazy Array Support

**DelayedArray Integration** (`R/as_delayed_array.R`):
- Conditional support for DelayedArray for memory-efficient operations
- Backend-specific seed implementations: MatrixBackendSeed
- Registered in `.onLoad()` if DelayedArray available
- Supports as_delayed_array() conversion for all backends

**Custom delarr wrapper** (`R/as_delarr.R`):
- Alternative lazy array interface via `delarr` package
- Provides custom wrapper for remote GitHub package

## Authentication & Access

**No Authentication Required:**
- Local file-based formats: NIfTI, HDF5, Zarr (local), matrix, in-memory
- Cloud storage (S3, GCS, Azure): Via Rarr package's native cloud support (no API keys in fmridataset)
- BIDS: Local filesystem access only

## Environment Variables

**Not Used:**
- No environment variables for configuration or authentication
- All settings via R options or function parameters
- Configuration files: YAML, JSON, or legacy DCF format

---

*Integration audit: 2026-01-22*
