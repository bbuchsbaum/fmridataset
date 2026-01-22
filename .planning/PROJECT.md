# fmridataset CRAN-Ready Milestone

## What This Is

An R package providing a unified S3 class (`fmri_dataset`) for representing fMRI data from multiple sources — NIfTI files, HDF5, Zarr, in-memory matrices, and BIDS datasets. This milestone focuses on making the package pass `R CMD check --as-cran` with high test coverage and resolving the Zarr backend viability question.

## Core Value

Backend-agnostic fMRI data access: one API works across all storage formats, with lazy loading, chunked iteration, and multi-subject support.

## Requirements

### Validated

Existing capabilities from the codebase:

- ✓ Multiple backend support (NIfTI, H5, matrix, latent, study) — existing
- ✓ Lazy loading via delarr/DelayedArray integration — existing
- ✓ Chunked iteration for memory-efficient processing — existing
- ✓ Temporal structure handling via sampling_frame — existing
- ✓ Multi-subject group operations — existing
- ✓ Series selection with voxel/coordinate/ROI support — existing
- ✓ Backend registry for extensibility — existing

### Active

- [ ] Pass `R CMD check --as-cran` with 0 errors, 0 warnings, 0 notes
- [ ] Achieve 80%+ overall test coverage (currently 62.75%)
- [ ] Investigate Zarr backend viability (Rarr dependency, cloud path support)
- [ ] Fix H5 backend resource leak issues identified in CONCERNS.md
- [ ] Fix unstated test dependencies (DelayedArray, rhdf5, etc.)
- [ ] Remove non-standard top-level files or add to .Rbuildignore
- [ ] Fix Rd cross-reference for sampling_frame link
- [ ] Fix undefined global function `generate_all_golden_data`
- [ ] Add missing vignette dependencies to Suggests

### Out of Scope

- Actual CRAN submission — just achieving readiness for when dependencies are on CRAN
- Streaming I/O for Zarr — complexity beyond current scope
- Sparse matrix support — would require significant backend changes
- New feature development — focus is quality, not features

## Context

**Dependency Status:**
- neuroim2: On CRAN ✓
- fmrihrf: On CRAN ✓
- delarr: Going to CRAN soon
- bidser: Going to CRAN soon
- fmristore: Going to CRAN soon

**Current R CMD check Status:**
- 0 errors
- 1 warning (unstated test dependencies)
- 5 notes (hidden files, non-standard files, undefined global, Rd link, vignette deps)

**Test Coverage by Area:**
- Well covered (>80%): matrix_backend, storage_backend, group operations, series selector
- Needs work (20-70%): h5_backend (26%), data_access (67%), nifti_backend (74%)
- Critical gaps (<10%): zarr_backend (5%), as_delayed_array (6-10%)

**Zarr Situation:**
- Depends on Rarr package from Bioconductor
- Bioconductor packages complicate CRAN submission
- Cloud paths (S3, GCS, Azure) are untested
- Need to determine: viable as optional feature, or remove entirely?

## Constraints

- **Dependencies**: Cannot submit to CRAN until delarr, bidser, fmristore are accepted
- **Zarr/Rarr**: Bioconductor dependency may require Zarr to be Suggests-only
- **Backward compatibility**: Existing API must remain stable for current users

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Target 80% coverage | Balance thoroughness with pragmatism | — Pending |
| Investigate Zarr fully | User wants cloud-native support if viable | — Pending |
| Fix check issues before adding coverage | Unblocks CI/CD quality gates | — Pending |

---
*Last updated: 2026-01-22 after initialization*
