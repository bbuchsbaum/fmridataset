# Codebase Concerns

**Analysis Date:** 2026-01-22

## Tech Debt

**Resource Leak Risk in H5 Backend Metadata Retrieval:**
- Issue: `backend_get_metadata.h5_backend()` at `R/h5_backend.R:408-409` opens an H5NeuroVec temporarily using `on.exit(close(first_h5))`, but if metadata extraction fails after the handle is opened, the handle remains open. The error condition from metadata operations (like `space(h5_obj)`) could bypass the `on.exit` cleanup.
- Files: `R/h5_backend.R` (lines 402-427)
- Impact: H5 file handles may leak in error scenarios, causing "file already open" errors on repeated calls or preventing file deletion on Windows
- Fix approach: Wrap the entire metadata extraction in `tryCatch()` with explicit `on.exit()` cleanup before any operations that could fail

**Potential Resource Leak in H5 Data Reading:**
- Issue: `backend_get_data.h5_backend()` at `R/h5_backend.R:336-397` loads H5NeuroVec objects on demand without `on.exit()` protection. If an error occurs during `neuroim2::series()` extraction (lines 371-377), the H5 objects are never closed, even though lines 381-384 attempt cleanup.
- Files: `R/h5_backend.R` (lines 336-397)
- Impact: Repeated read errors will leak H5 file handles, degrading performance and potentially causing resource exhaustion
- Fix approach: Use `on.exit()` wrapper around the entire block before any data access operations, not just conditional cleanup after-the-fact

**H5 Dimension Query Loops Open/Close Repeatedly:**
- Issue: `backend_get_dims.h5_backend()` at `R/h5_backend.R:221-227` opens and closes an H5 file for every source file to sum time dimensions. For datasets with 50+ runs, this means 50+ file open/close cycles during dimension query alone.
- Files: `R/h5_backend.R` (lines 206-257)
- Impact: Significant performance degradation for multi-run studies; cumulative I/O overhead; potential file locking issues on network storage
- Fix approach: Cache dimension query results in backend state; or use H5 metadata-only access if available

**Cache Management Inconsistency:**
- Issue: `R/data_access.R` initializes caches with configurable size (lines 82-84), but cache resizing after package load is explicitly prevented (lines 231-232). Users setting `options(fmridataset.cache_max_mb = X)` after package load get a warning instead of dynamic resizing.
- Files: `R/data_access.R` (lines 75-100, 225-232)
- Impact: Users cannot adapt cache strategy to runtime conditions; memory pressure situations cannot be mitigated without restarting R
- Fix approach: Implement cache size adjustment at runtime, or provide cache flush/reset utilities

## Known Bugs

**Backend Method Resolution Issue (In Progress Fix):**
- Symptoms: `validate_backend()` may fail to detect properly implemented S3 methods due to namespace/environment lookup issues
- Files: `R/storage_backend.R` (lines 127-142) - currently modified to use `utils::getS3method()` instead of `exists()`
- Trigger: When a backend class is defined in a different package or namespace environment
- Workaround: Ensure backend classes are defined in the caller's environment or explicitly register them
- Fix approach: The pending change (R/storage_backend.R git diff) already addresses this by switching from `exists()` to `utils::getS3method()`

**Zarr Backend Dimension Validation:**
- Symptoms: `backend_open.zarr_backend()` expects exactly 4D arrays (x, y, z, time) but error message doesn't clarify dimension ordering
- Files: `R/zarr_backend.R` (lines 126-134)
- Trigger: When Zarr arrays with different dimension orders are provided
- Workaround: Ensure source Zarr arrays follow (spatial[1:3], time) ordering before backend creation
- Fix approach: Add dimension reordering utility or document expected ordering more explicitly

## Security Considerations

**File Path Validation in Dummy Mode:**
- Risk: `dummy_mode=TRUE` in `nifti_backend()` and `latent_backend()` bypasses file existence checks, allowing downstream operations on non-existent paths
- Files: `R/nifti_backend.R` (lines 34-44, 67-76), `R/latent_backend.R` (lines 43-60)
- Current mitigation: dummy_mode is documented as testing-only feature; not intended for production use
- Recommendations: Add runtime warnings when dummy_mode is active; validate that dummy_mode data produces sensible dimensions before allowing data access

**External Package Dependency Chain:**
- Risk: Package depends on remote packages via Remotes field (DESCRIPTION lines 56-60): `delarr`, `fmrihrf`, `fmristore`, `bidser`. If any remote package is compromised, all users are affected.
- Files: `DESCRIPTION`, imports throughout (R/*.R)
- Current mitigation: Use HTTPS for GitHub remotes; rely on GitHub security
- Recommendations: Implement package integrity verification at install time; document dependency chain in security advisories

## Performance Bottlenecks

**Repeated NIfTI Metadata Extraction:**
- Problem: Every `backend_get_metadata.nifti_backend()` call re-reads header information and reconstructs NeuroSpace objects
- Files: `R/nifti_backend.R` (lines 446-530)
- Cause: Caching only happens within a single function call (line 529); metadata is not cached at backend initialization
- Improvement path: Cache metadata at `backend_open()` time; implement LRU cache for frequently accessed files

**Study Backend Spatial Dimension Validation:**
- Problem: `study_backend()` validates identical spatial dimensions across all backends by opening each and calling `backend_get_dims()`
- Files: `R/study_backend.R` (lines 58-72)
- Cause: No cached dimension information; full backend queries for validation during construction
- Improvement path: Pre-cache backend dimensions; use lazy validation pattern or deferred checks

**Data Type Conversions in Series Access:**
- Problem: 70+ `as.numeric()`, `as.integer()`, `as.logical()`, `as.character()` calls across codebase without validation
- Files: Multiple files use unsafe conversions
- Cause: Implicit type coercion can be slow and hide bugs in boundary cases
- Improvement path: Replace with explicit, validated type conversion using assertthat predicates

## Fragile Areas

**Series Selector Class System:**
- Files: `R/series_selector.R` (436 lines)
- Why fragile: Complex subsetting logic with multiple code paths (numeric indices, logical masks, character selection). Edge cases around voxel index remapping when masks change.
- Safe modification: Add comprehensive boundary condition tests; validate that selected voxels remain within mask after subsetting
- Test coverage: `test_series_selector.R` exists but coverage may not include all edge cases involving partial mask selection

**Latent Backend Data Reconstruction:**
- Files: `R/latent_backend.R` (454 lines)
- Why fragile: Reconstructs full voxel data from basis functions + loadings. Numerical stability depends on basis orthogonality and loading precision. No validation of reconstruction accuracy.
- Safe modification: Add tolerance checks for basis orthogonality; implement reconstruction quality metrics; add warnings if reconstructed variance deviates from original
- Test coverage: `test_latent_backend.R` and `test_latent_performance.R` exist but focus on API compatibility, not numerical accuracy

**Study Backend Mask Intersection Logic:**
- Files: `R/study_backend.R` (lines 74-90)
- Why fragile: Supports both "identical" and "intersect" mask validation modes. "intersect" mode silently reduces usable voxels per subject, which could lead to per-subject data corruption if not carefully tracked.
- Safe modification: Add clear warnings when mask intersection occurs; log reduced voxel count per subject; add validation that intersection result is non-empty
- Test coverage: `test_study_backend.R` and `test_study_backend_memory.R` test basic functionality but not mask intersection edge cases

**Backend Registry Pattern Matching:**
- Files: `R/backend_registry.R` (431 lines)
- Why fragile: Infers backend type from file extensions using pattern matching (e.g., `\\.lv\\.h5$`). Case sensitivity, dot escaping, and double-extension handling could introduce subtle bugs.
- Safe modification: Use explicit MIME type detection or magic bytes; validate actual file format before assuming backend type
- Test coverage: No dedicated test file for backend_registry patterns

## Scaling Limits

**H5 Backend Open/Close Cycles:**
- Current capacity: Efficient for ~10 runs; degradation observed with 50+ runs
- Limit: File I/O becomes bottleneck when querying dimensions for large multi-run studies
- Scaling path: Implement H5 index files or metadata sidecar files to avoid per-run queries; cache dimension information globally

**Memory Cache LRU Eviction:**
- Current capacity: Default 512MB configured via option `fmridataset.cache_max_mb`
- Limit: Large datasets with many concurrent voxel accesses will experience high cache miss rates; LRU policy may not be optimal for sequential I/O
- Scaling path: Implement alternative eviction policies (e.g., LFU, segment-based); support tiered caching (memory + disk)

**Study Backend Lazy Evaluation:**
- Current capacity: StudyBackendSeed pattern supports lazy initialization, but entire study mask must fit in memory
- Limit: Cannot support studies with >1 million voxels per mask
- Scaling path: Implement virtual mask representation; support mask streaming or tiling

## Dependencies at Risk

**Required dependency on custom remote packages:**
- Risk: Package fails to install if GitHub is unreachable or if remote packages have unresolved dependencies
- Impact: Blocks entire package installation; cascades to downstream packages
- Migration plan: Mirror critical dependencies (delarr, fmristore) to CRAN; fall back to compiled snapshot versions

**Optional dependency on Rarr (BiocManager):**
- Risk: Zarr backend requires Rarr which depends on BiocManager for installation
- Impact: Windows users may experience installation failures; Bioconductor release cycles may lag critical updates
- Migration plan: Evaluate alternative Zarr implementations that don't require BiocManager; provide pre-built binaries

**Weak version pinning in DESCRIPTION:**
- Risk: `neuroim2`, `delarr`, `fmrihrf` are available on GitHub but not CRAN; no version constraints specified
- Impact: Upstream changes break compatibility without warning
- Migration plan: Add version constraints (e.g., `neuroim2 (>= 0.4.0)`); establish semantic versioning policy with dependencies

## Missing Critical Features

**No Sparse Matrix Support:**
- Problem: Large studies often generate sparse voxel selection masks, but all backends assume dense voxel arrays
- Blocks: Memory-efficient processing of highly masked datasets; efficient partial-voxel-set reanalysis
- Fix: Add sparse matrix variants of backends; implement efficient sparse indexing in `backend_get_data()`

**No Streaming I/O for Zarr:**
- Problem: Zarr backend must fully define Zarr store structure at backend creation time; cannot append data
- Blocks: Incremental data collection workflows; online analysis of streaming fMRI data
- Fix: Implement append-mode for Zarr backend; add streaming API

**No Explicit Data Validation Framework:**
- Problem: No built-in checks for data sanity (e.g., voxel mean across time, temporal variance, expected value ranges)
- Blocks: Early detection of data corruption, scanner errors, or file read failures
- Fix: Add optional validation pass at dataset construction; provide per-voxel quality metrics

## Test Coverage Gaps

**H5 Backend Resource Cleanup:**
- What's not tested: Behavior when H5 files become unavailable after backend creation; repeated error conditions and recovery
- Files: `R/h5_backend.R`, test file `test_h5_backend.R` (if exists)
- Risk: Handle leaks in production scenarios with transient I/O errors; tests only cover happy path
- Priority: High

**Zarr Backend Remote Storage Paths:**
- What's not tested: S3://, gs://, and Azure:// remote Zarr store paths; authentication and timeouts
- Files: `R/zarr_backend.R`, test file `test_zarr_backend.R`
- Risk: Cloud-native workflows will fail in untested ways; no integration tests with real cloud storage
- Priority: Medium

**Study Backend Mask Combination Modes:**
- What's not tested: Edge cases where subject masks have significant differences; intersection mode with very small overlaps
- Files: `R/study_backend.R`, test file `test_fmri_study_dataset.R`
- Risk: Silent data loss or corruption when combining incompatible subject masks
- Priority: Medium

**Error Recovery and Retry Logic:**
- What's not tested: Transient I/O errors, network timeouts, file locking scenarios; recovery behavior
- Files: Multiple backend files with `tryCatch()` handlers
- Risk: Unpredictable behavior in cloud/network storage scenarios; no guidance on retry policies
- Priority: Medium

**Latent Backend Numerical Stability:**
- What's not tested: Precision of basis function orthogonality; reconstruction accuracy for extreme coefficient values
- Files: `R/latent_backend.R`, test file `test_latent_backend.R`
- Risk: Silent numerical errors in latent space reconstructions; undetected coefficient overflow/underflow
- Priority: Medium

---

*Concerns audit: 2026-01-22*
