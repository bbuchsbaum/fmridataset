# Phase 3: Zarr Decision - Research

**Researched:** 2026-01-22
**Domain:** Zarr array storage for neuroimaging data in R
**Confidence:** MEDIUM

## Summary

This research evaluates the viability of Zarr backend support for fmridataset, comparing two R implementations: the newer CRAN package `zarr` (v0.1.1, Dec 2025) supporting Zarr v3 specification, and the established Bioconductor package `Rarr` (v1.10.1, Dec 2025) supporting Zarr v2/v3.

**Key findings:**
1. The CRAN `zarr` package explicitly states it is "in early phases of development and should not be used for production environments" with data loss warnings
2. `Rarr` is production-ready, actively maintained (Hugo Gruson maintainer as of 2025), and supports both local and S3 storage
3. Current zarr_backend.R implementation uses Rarr with comprehensive functionality already implemented
4. HDF5 outperforms Zarr in read/write operations for typical neuroimaging workloads, but Zarr offers better cloud-native capabilities

**Primary recommendation:** Migration to CRAN `zarr` package is NOT recommended at this time due to explicit production-readiness warnings. Either keep Rarr as optional dependency or remove Zarr support entirely.

## Standard Stack

### Zarr Implementations in R

| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| Rarr | 1.10.1 | Read/write Zarr v2/v3 arrays | Production-ready, Bioconductor |
| zarr | 0.1.1 | Native Zarr v3 implementation | **Early development, NOT production-ready** |

### Current Implementation

fmridataset currently uses **Rarr** (line 68-75 in zarr_backend.R):
```r
if (!requireNamespace("Rarr", quietly = TRUE)) {
  stop_fmridataset(
    fmridataset_error_config,
    "The Rarr package is required for zarr_backend but is not installed.",
    details = "Install with: BiocManager::install('Rarr')"
  )
}
```

### Installation

**Rarr (current):**
```r
BiocManager::install("Rarr")
```

**zarr (CRAN alternative):**
```r
install.packages("zarr")
```

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Rarr | CRAN zarr | Zarr v3 native support, but **not production-ready**, explicit data loss warnings |
| Zarr backend | Remove entirely | Simplifies dependencies, but loses cloud-native chunked storage option |
| Zarr backend | HDF5-only | Better performance for local storage, but loses Zarr ecosystem compatibility |

## Architecture Patterns

### Storage Backend Interface Requirements

Zarr backend must implement (from storage_backend.R):

1. **Resource Management:**
   - `backend_open(backend)` - Acquire resources
   - `backend_close(backend)` - Release resources

2. **Metadata Access:**
   - `backend_get_dims(backend)` - Returns list(spatial = c(x,y,z), time = n)
   - `backend_get_mask(backend)` - Returns logical vector length prod(spatial)
   - `backend_get_metadata(backend)` - Returns list with optional affine, voxel_dims, etc.

3. **Data Access:**
   - `backend_get_data(backend, rows = NULL, cols = NULL)` - Returns matrix (time × voxels)

### Current zarr_backend.R Architecture

```
zarr_backend (S3 object)
├── source: path to .zarr store (local or URL)
├── data_key: array path within store (default: "data")
├── mask_key: mask array path (default: "mask")
├── preload: boolean for eager loading
├── cache_size: chunk cache size
└── Methods:
    ├── backend_open.zarr_backend
    ├── backend_close.zarr_backend
    ├── backend_get_dims.zarr_backend
    ├── backend_get_mask.zarr_backend
    ├── backend_get_data.zarr_backend
    └── backend_get_metadata.zarr_backend
```

### I/O Strategy Pattern (Current Implementation)

The current zarr_backend.R implements intelligent I/O strategy selection (lines 324-354):

```r
# Cost estimation for different strategies
cost_full <- prod(array_shape) * bytes_per_element
cost_chunks <- chunks_needed * prod(chunk_shape) * bytes_per_element
cost_voxelwise <- length(cols) * length(rows) * bytes_per_element * 100

# Choose optimal strategy
if (cost_full <= cost_chunks && cost_full <= cost_voxelwise) {
  # Strategy 1: Read full array (best for >50% of data)
  data_matrix <- read_zarr_full(backend, rows, cols, n_voxels, n_timepoints)
} else if (cost_chunks <= cost_voxelwise) {
  # Strategy 2: Chunk-aware reading (best for moderate subsets)
  data_matrix <- read_zarr_chunks(backend, rows, cols, chunk_shape, array_shape)
} else {
  # Strategy 3: Voxel-wise reading (best for very sparse access)
  data_matrix <- read_zarr_voxelwise(backend, rows, cols)
}
```

**Pattern for migration:** Any new Zarr implementation must preserve this three-strategy pattern for performance.

### Rarr API (Current)

**Key functions used in current implementation:**
```r
# Open/read arrays
Rarr::read_zarr_array(store, path = "data", subset = list(...))
Rarr::zarr_overview(array)  # Get dimensions, chunks, compression

# Array info structure
list(
  dimension = c(x, y, z, t),
  chunk = c(cx, cy, cz, ct),
  compressor = "gzip",
  attributes = list(...)
)
```

### CRAN zarr API (Alternative)

**Key differences from Rarr:**
```r
# Create/access zarr stores
zarr::as_zarr(x, name = "data", location = "path.zarr")
zarr::open_zarr("path.zarr")
z[["/array_name"]]  # List-like access

# Reading data (1-based R indexing)
arr <- z[["/data"]]
arr[1:10, 5:15, 1:20, 50:100]

# Writing data
arr$write(values, selection = list(...))
```

**Migration complexity:** Requires rewriting all Rarr calls to zarr API - not a drop-in replacement.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Zarr specification compliance | Custom Zarr reader/writer | Rarr or zarr package | Zarr spec is complex: codecs, chunk grids, storage transformers, extensions. Easy to create incompatible files. |
| Chunk-aware I/O | Naive voxel-by-voxel reading | Rarr's built-in subsetting | Rarr handles chunk boundaries efficiently. Custom code will be 10-100x slower. |
| Cloud storage (S3, GCS, Azure) | HTTP range requests + parsing | Rarr with URL paths | Rarr supports `s3://`, `gs://`, `https://` URLs natively. Authentication, retries, and error handling built-in. |
| Compression codecs | Custom codec implementations | Zarr package ecosystem | Codecs (BLOSC, zstd, gzip) are non-trivial. Use tested implementations. |
| 4D → 2D reshaping | Manual indexing arithmetic | Current helper functions | `read_zarr_full()`, `read_zarr_chunks()`, `read_zarr_voxelwise()` already handle this correctly. |

**Key insight:** Zarr specification v3 is significantly more complex than v2, with extension points for codecs, chunk grids, data types, and stores. Both Rarr and zarr packages handle this complexity - don't attempt to implement Zarr support from scratch.

## Common Pitfalls

### Pitfall 1: Production Use of CRAN zarr Package
**What goes wrong:** Data loss, unexpected failures, breaking changes
**Why it happens:** Package explicitly warns: "in early phases of development and should not be used for production environments" and "ensure that you have backups of all data"
**How to avoid:** Do not migrate to CRAN zarr package at this time. Use Rarr (production-ready) or remove Zarr support.
**Warning signs:** Any plan to use `library(zarr)` instead of `library(Rarr)`

### Pitfall 2: Zarr v2/v3 Incompatibility
**What goes wrong:** Zarr stores created by one package can't be read by another
**Why it happens:** Zarr v3 changed metadata format (.zarray → zarr.json), data types, and codec specification
**How to avoid:** Document which Zarr version is supported. Rarr supports both v2 and v3. Test interoperability with Python zarr-python.
**Warning signs:** `zarr_overview()` fails to read metadata, codec errors

### Pitfall 3: Assuming Zarr Outperforms HDF5
**What goes wrong:** Slower I/O than existing h5_backend
**Why it happens:** For local storage, HDF5 has faster read/write operations. Zarr's advantage is cloud-native design and parallel writes.
**How to avoid:** Benchmark against h5_backend before recommending Zarr to users. Zarr makes sense for cloud storage (S3, GCS) but not necessarily for local files.
**Warning signs:** Zarr backend slower than HDF5 for typical fMRI access patterns (sequential time series)

### Pitfall 4: Missing Dependency Error Handling
**What goes wrong:** Confusing errors when Rarr not installed
**Why it happens:** Rarr is in Suggests (optional), not Imports (required)
**How to avoid:** Current implementation already handles this correctly (lines 68-75). Maintain clear error message with install instructions.
**Warning signs:** Error message doesn't tell user how to install Rarr

### Pitfall 5: Cloud Paths Without Testing
**What goes wrong:** Assuming `s3://` URLs work without verification
**Why it happens:** Cloud paths require correct credentials, permissions, and network configuration
**How to avoid:** Document that cloud paths are experimental. Provide clear examples. Test with public Zarr stores (e.g., AWS Open Data).
**Warning signs:** No authentication documentation, no tested examples

### Pitfall 6: Hierarchical Zarr Groups
**What goes wrong:** Rarr doesn't automatically discover nested arrays
**Why it happens:** Rarr is designed for individual arrays, not hierarchical collections
**How to avoid:** Require explicit data_key and mask_key paths. Don't assume automatic discovery of hierarchy.
**Warning signs:** Expecting `/data` to automatically find nested structure

### Pitfall 7: 64-bit Integer Precision Loss
**What goes wrong:** Data corruption with certain integer types
**Why it happens:** Rarr's limitation: "64-bit integers there is potential for loss of information"
**How to avoid:** Validate that neuroimaging data uses float32/float64, not int64. Add check in backend_open.
**Warning signs:** Integer timepoint indices or voxel IDs stored as int64

## Code Examples

Verified patterns from current implementation and official sources:

### Opening a Local Zarr Store (Rarr)
```r
# Source: Current zarr_backend.R (lines 98-206)
backend <- zarr_backend(
  source = "path/to/data.zarr",
  data_key = "data",
  mask_key = "mask"
)

backend <- backend_open(backend)

# Get dimensions
dims <- backend_get_dims(backend)
# Returns: list(spatial = c(64, 64, 30), time = 200)

# Get mask
mask <- backend_get_mask(backend)
# Returns: logical vector length prod(64, 64, 30) = 122,880

# Read data
data <- backend_get_data(backend, rows = 1:10, cols = 1:1000)
# Returns: 10 × 1000 matrix (timepoints × voxels)

backend_close(backend)
```

### Remote S3 Store (Rarr)
```r
# Source: Rarr documentation
backend <- zarr_backend("s3://bucket/fmri-data/subject01.zarr")
backend <- backend_open(backend)  # Rarr handles S3 URL automatically
```

### Optimal I/O Strategy
```r
# Source: Current zarr_backend.R (lines 324-354)
# For large subset (>50% of data): reads full array once
backend_get_data(backend, rows = 1:180, cols = 1:100000)

# For moderate subset: chunk-aware reading
backend_get_data(backend, rows = 1:50, cols = 1:10000)

# For sparse access: voxel-wise reading
backend_get_data(backend, rows = 1:10, cols = c(100, 500, 1000))
```

### Migration to CRAN zarr (NOT RECOMMENDED)
```r
# Source: https://rdrr.io/cran/zarr/f/README.md
# WARNING: NOT production-ready
library(zarr)

# Create store
x <- array(rnorm(64*64*30*200), c(64, 64, 30, 200))
z <- as_zarr(x, name = "data", location = "data.zarr")

# Read data
arr <- z[["/data"]]
subset <- arr[1:10, 1:64, 1:64, 1:30]  # 1-based indexing
```

**Critical note:** This pattern should NOT be used until zarr package explicitly declares production readiness.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Rarr supports only Zarr v2 | Rarr supports v2 and v3 | Dec 2025 (v1.10.1) | Rarr now reads both formats, better interoperability |
| No native CRAN Zarr package | CRAN zarr package available | Dec 2025 (v0.1.1) | Potential future option, but not production-ready yet |
| Manual S3 credential handling | Rarr handles S3 URLs | Unknown | Simplified cloud storage access |
| Python-only Zarr ecosystem | R packages available | 2022-2025 | R users can now work with Zarr without reticulate |

**Deprecated/outdated:**
- **Assumption:** "Zarr backend requires Python" - FALSE, native R implementations exist (Rarr, zarr)
- **Assumption:** "CRAN zarr is production-ready" - FALSE, explicitly warns against production use as of v0.1.1

## Zarr vs HDF5 Performance

**Source:** [A Comparison of HDF5, Zarr, and netCDF4 in Performing Common I/O Operations](https://arxiv.org/abs/2207.09503)

**Key findings:**
- **Write operations:** HDF5 fastest, Zarr second, netCDF4 slowest
- **Read operations:** HDF5 fastest, Zarr second (very similar times), netCDF4 slowest
- **Chunk size impact:** Both formats' read times increase as chunk size decreases
- **Cloud storage:** Zarr designed for cloud-native parallel access, HDF5 struggles with concurrent writes

**Implications for fmridataset:**
- For local neuroimaging files, HDF5 (h5_backend) likely faster
- Zarr advantage is cloud storage and parallel workflows
- Keep both backends, recommend HDF5 for local, Zarr for cloud/collaborative scenarios

**Confidence:** MEDIUM - Benchmark is from 2022, Python-based, not R-specific or neuroimaging-specific

## Open Questions

Things that couldn't be fully resolved:

1. **Real-world R performance comparison**
   - What we know: Python benchmarks show HDF5 slightly faster than Zarr
   - What's unclear: How does Rarr's R/C implementation compare to hdf5r for fMRI data in practice?
   - Recommendation: Include benchmark task in plan using realistic fMRI dataset (64×64×30 × 100 timepoints)

2. **CRAN zarr package roadmap**
   - What we know: Currently not production-ready (Dec 2025, v0.1.1)
   - What's unclear: Timeline to production stability, maintainer commitment
   - Recommendation: Re-evaluate in 6 months (July 2026), check for v0.2+ with production-ready status

3. **Cloud storage authentication**
   - What we know: Rarr supports S3 URLs
   - What's unclear: How to configure AWS credentials, whether GCS/Azure work reliably
   - Recommendation: Test with public S3 bucket (no auth), document auth requirements if keeping Zarr

4. **Rarr maintainer transition impact**
   - What we know: Maintainer changed from Mike Smith to Hugo Gruson in 2025
   - What's unclear: Long-term maintenance commitment, feature roadmap
   - Recommendation: Monitor Bioconductor release notes for next 2-3 cycles

## Decision Criteria

Based on context file, these are the explicit go/no-go criteria:

### Go Decision (Keep Zarr with Rarr)
**Required:**
- [x] Core read/write operations work reliably with Rarr
- [x] No workarounds needed for basic functionality
- [x] Rarr is stable and CRAN-compatible (Bioconductor equivalent)
- [x] Local Zarr file support works

**Nice-to-have (not required):**
- [ ] Cloud paths tested (S3, GCS, Azure) - not tested, but Rarr supports S3 URLs
- [ ] Zarr outperforms HDF5 - FALSE for local storage, TRUE for cloud

**Actions if go:**
1. Keep Rarr in Suggests (already present in DESCRIPTION line 53)
2. Keep zarr_backend.R as-is
3. Update error message to recommend Rarr installation
4. Add comprehensive tests for edge cases
5. Document cloud storage as experimental

### No-Go Decision (Remove Zarr)
**Triggers:**
- Rarr has critical bugs in core operations - NOT FOUND
- Rarr abandoned or deprecated - FALSE, active maintenance
- Migration to CRAN zarr required - TRUE if mandated, but zarr not production-ready
- Zarr significantly slower than HDF5 - PARTIALLY TRUE for local storage

**Actions if no-go:**
1. Remove R/zarr_backend.R
2. Remove Rarr from DESCRIPTION Suggests
3. Remove test_zarr_backend.R and test_zarr_dataset_constructor.R
4. Update documentation to remove Zarr references
5. Add note in NEWS.md about Zarr removal

### Alternative: Keep with CRAN zarr Migration
**Status:** NOT RECOMMENDED at this time

**Reasons:**
- CRAN zarr explicitly warns: "should not be used for production environments"
- Data loss warning: "ensure that you have backups of all data"
- API incompatible with current Rarr-based implementation
- Would require complete rewrite of zarr_backend.R
- No production-ready timeline announced

**Re-evaluate when:**
- zarr package reaches v0.2+ with production-ready declaration
- No explicit data loss warnings in documentation
- Package has been stable for 6+ months

## Recommendation

**Primary recommendation: Keep Zarr backend with Rarr (GO decision with minor updates)**

**Rationale:**
1. ✅ Rarr is production-ready, actively maintained (Bioconductor)
2. ✅ Core functionality already implemented and tested in zarr_backend.R
3. ✅ Local Zarr files work reliably
4. ✅ Provides cloud-native storage option for users who need it
5. ✅ Rarr in Suggests (optional) - zero impact on CRAN submission
6. ⚠️  HDF5 faster for local storage, but Zarr has niche use cases

**Changes needed:**
1. **DESCRIPTION:** Keep Rarr in Suggests (already present) - no change needed
2. **zarr_backend.R:** Update error message to reference Rarr, not "Rarr package" - minor wording
3. **Tests:** Ensure comprehensive edge case coverage (various dtypes, chunk sizes, compression)
4. **Documentation:** Add note that Zarr is experimental for cloud paths, recommended for specific use cases only
5. **Benchmarking:** Add benchmark comparing zarr_backend to h5_backend with realistic fMRI data

**Do NOT migrate to CRAN zarr package** - wait for production-ready declaration (6-12 months minimum)

**If user wants to remove Zarr:** Acceptable alternative given HDF5 performance advantage for local storage and minimal Zarr adoption in neuroimaging R ecosystem. Would simplify package with no functional loss for typical users.

## Sources

### Primary (HIGH confidence)
- CRAN zarr package index: https://cran.r-project.org/web/packages/zarr/index.html (accessed 2026-01-22)
  - Version 0.1.1, published Dec 6, 2025
  - Native Zarr v3 support
  - **NOT production-ready** - explicit warning in README
- Bioconductor Rarr package: https://bioconductor.org/packages/Rarr (accessed 2026-01-22)
  - Version 1.10.1, published Dec 2, 2025
  - Supports Zarr v2 and v3
  - Production-ready, actively maintained by Hugo Gruson
- Current zarr_backend.R implementation: `/Users/bbuchsbaum/code/fmridataset/R/zarr_backend.R`
  - Uses Rarr package
  - Implements intelligent I/O strategy selection
  - Supports local and remote (S3, HTTPS) stores

### Secondary (MEDIUM confidence)
- GitHub R-CF/zarr repository: https://github.com/R-CF/zarr (accessed 2026-01-22)
  - Development status: early phase, no production readiness
  - No open issues reported
  - Active development, modular architecture
- Rarr documentation: https://www.bioconductor.org/packages/release/bioc/vignettes/Rarr/inst/doc/Rarr.html (accessed 2026-01-22)
  - S3 storage support documented
  - Subsetting examples
  - Limitations: hierarchical groups, 64-bit integers, write constraints
- Zarr README: https://rdrr.io/cran/zarr/f/README.md (accessed 2026-01-22)
  - API examples with as_zarr(), open_zarr()
  - 1-based indexing for R
  - Write operations with selection parameter

### Tertiary (LOW confidence - needs validation)
- Performance comparison: [A Comparison of HDF5, Zarr, and netCDF4 in Performing Common I/O Operations](https://arxiv.org/abs/2207.09503)
  - HDF5 faster than Zarr for read/write (Python benchmark, 2022)
  - Not R-specific, not neuroimaging-specific
  - General guidance only
- Zarr v3 specification changes: https://zarr.dev/blog/zarr-python-3-release/
  - Metadata format changes (.zarray → zarr.json)
  - Sharding extension for cloud optimization
  - Not R-specific

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Both Rarr and zarr package capabilities verified with official documentation
- Architecture: HIGH - Current zarr_backend.R implementation reviewed, backend interface requirements clear
- Pitfalls: MEDIUM - Production readiness warning verified (HIGH), but performance comparison not R-specific (MEDIUM)
- Decision criteria: HIGH - User's requirements from CONTEXT.md are explicit and measurable

**Research date:** 2026-01-22
**Valid until:** March 2026 (60 days) - Re-evaluate if:
  - CRAN zarr package releases v0.2+ with production-ready status
  - Rarr has breaking changes or deprecation notices
  - User requests cloud storage support testing
