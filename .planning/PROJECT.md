# fmridataset

## What This Is

An R package providing a unified S3 class (`fmri_dataset`) for representing fMRI data from multiple sources — NIfTI files, HDF5, Zarr, in-memory matrices, and BIDS datasets. The package is CRAN-ready (v0.9.0) and passes R CMD check with 0 errors.

## Core Value

Backend-agnostic fMRI data access: one API works across all storage formats, with lazy loading, chunked iteration, and multi-subject support.

## Current State

**Version:** 0.9.0 (CRAN-Ready)
**Shipped:** 2026-01-22

**R CMD Check Status:**
- 0 errors
- 1 warning (non-CRAN dependencies: delarr, bidser, fmristore)
- 1 note (new submission)

**Test Coverage:** 73.3% (1991 tests passing)
- Well covered: zarr_backend (94.6%), as_delayed_array_dataset (100%), matrix_backend, storage_backend
- Documented gaps: h5_backend (30.1% - S4 mocking limitation)

**Blocker for CRAN submission:** Waiting for delarr, bidser, fmristore to be accepted on CRAN.

## Requirements

### Validated

Shipped in v0.9.0:
- ✓ Pass R CMD check --as-cran with 0 errors — v0.9.0
- ✓ Zarr backend viability resolved (migrated to CRAN zarr, experimental) — v0.9.0
- ✓ H5 backend resource leaks fixed — v0.9.0
- ✓ Test dependencies properly declared — v0.9.0
- ✓ 73.3% test coverage with documented gaps — v0.9.0

Existing capabilities:
- ✓ Multiple backend support (NIfTI, H5, Zarr, matrix, latent, study)
- ✓ Lazy loading via delarr/DelayedArray integration
- ✓ Chunked iteration for memory-efficient processing
- ✓ Temporal structure handling via sampling_frame
- ✓ Multi-subject group operations
- ✓ Series selection with voxel/coordinate/ROI support
- ✓ Backend registry for extensibility

### Active

**v0.9.1 Documentation Quality:**
- [ ] Vignettes have executable examples (fix eval=FALSE where appropriate)
- [ ] Vignette content matches current API
- [ ] Vignettes have clear, user-centric explanations
- [ ] pkgdown site rebuilt with updated content
- [ ] Roxygen2 docs are current and complete

### Out of Scope

- Actual CRAN submission — waiting for upstream dependencies
- Streaming I/O for Zarr — complexity beyond current scope
- Sparse matrix support — would require significant backend changes
- h5_backend S4 test fixtures — medium investment, optional feature

## Context

**Dependency Status:**
- neuroim2: On CRAN ✓
- fmrihrf: On CRAN ✓
- delarr: Pending CRAN
- bidser: Pending CRAN
- fmristore: Pending CRAN

**Tech Stack:**
- ~73,000 lines of R code
- S3/S4 hybrid class system
- DelayedArray integration for lazy evaluation
- Multiple storage backends (NIfTI, HDF5, Zarr, matrix)

## Constraints

- **Dependencies**: Cannot submit to CRAN until delarr, bidser, fmristore are accepted
- **Zarr**: v3-only (CRAN zarr package limitation), marked EXPERIMENTAL
- **Backward compatibility**: Existing API must remain stable for current users
- **Testing**: S4 mocking fundamentally impossible for neuroim2 objects

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Migrate to CRAN zarr | Pure CRAN dependency, working API | ✓ Zarr v3-only, EXPERIMENTAL |
| Accept 73% coverage | S4 mocking impossible for h5_backend | ✓ Documented limitation |
| No Bioconductor deps | Simplifies CRAN submission | ✓ rhdf5 -> hdf5r, Rarr -> zarr |
| Auto-open backends | Validation needs dims before open | ✓ Follows nifti pattern |
| Alphabetize DESCRIPTION | Maintainability with 24+ deps | ✓ Consistent ordering |
| on.exit() for H5 cleanup | Prevent resource leaks | ✓ 6 locations fixed |

## Current Milestone: v0.9.1 Documentation Quality

**Goal:** Ensure vignettes are well-written with executable examples, accurate API usage, clear explanations, and rebuild pkgdown site.

**Target features:**
- Vignette quality audit and fixes across all 7 vignettes
- Switch from eval=FALSE to executable examples where feasible
- Ensure content accuracy against current codebase
- Rebuild pkgdown site with polished documentation

---
*Last updated: 2026-01-23 after v0.9.1 milestone start*
