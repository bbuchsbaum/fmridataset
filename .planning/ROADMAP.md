# Roadmap: fmridataset CRAN-Ready

## Overview

Transform fmridataset from feature-complete to CRAN-ready by fixing R CMD check issues, resolving tech debt in H5 backend, deciding on Zarr viability, and achieving 80%+ test coverage. Five focused phases move from quick compliance wins through infrastructure fixes to comprehensive testing, culminating in final CRAN validation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: CRAN Quick Wins** - Fix immediate R CMD check issues
- [x] **Phase 2: Tech Debt** - Resolve H5 backend resource leaks
- [x] **Phase 3: Zarr Decision** - Investigate and decide on Zarr support
- [x] **Phase 4: Test Coverage** - Achieve 80%+ coverage across backends
- [x] **Phase 5: Final Validation** - Verify CRAN compliance

## Phase Details

### Phase 1: CRAN Quick Wins
**Goal**: R CMD check runs cleanly except for final validation requirements
**Depends on**: Nothing (first phase)
**Requirements**: CRAN-01, CRAN-02, CRAN-03, CRAN-04, CRAN-05, CRAN-06
**Success Criteria** (what must be TRUE):
  1. All test dependencies (DelayedArray, rhdf5, devtools, iterators, withr) are in Suggests field
  2. All vignette dependencies (microbenchmark, pryr) are in Suggests field
  3. `generate_all_golden_data` function is properly defined or removed
  4. All Rd documentation cross-references resolve correctly
  5. .planning directory and non-standard files are in .Rbuildignore
  6. R CMD check produces at most warnings (no errors, no notes except dependencies)
**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md - Add missing test and vignette dependencies to DESCRIPTION
- [x] 01-02-PLAN.md - Fix golden_data architecture and sampling_frame cross-reference
- [x] 01-03-PLAN.md - Update .Rbuildignore for non-standard files

### Phase 2: Tech Debt
**Goal**: H5 backend has proper resource management and storage_backend fix is committed
**Depends on**: Phase 1
**Requirements**: DEBT-01, DEBT-02, DEBT-03
**Success Criteria** (what must be TRUE):
  1. H5 backend metadata retrieval uses on.exit() protection in all error paths
  2. H5 backend data reading closes H5 handles even when errors occur
  3. storage_backend.R getS3method change is committed to repository
**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md - Add on.exit() resource protection to H5 backend functions
- [x] 02-02-PLAN.md - Commit storage_backend.R getS3method fix

### Phase 3: Zarr Decision
**Goal**: Zarr backend viability determined with go/no-go decision
**Depends on**: Phase 2
**Requirements**: ZARR-01, ZARR-02, ZARR-03
**Success Criteria** (what must be TRUE):
  1. Rarr package stability and CRAN compatibility is documented
  2. Zarr cloud paths (S3, GCS, Azure) are tested with results documented
  3. Go/no-go decision is made: either Zarr moves to Suggests-only or is removed
  4. If keeping Zarr: DESCRIPTION updated with Rarr as Suggests
  5. If removing Zarr: zarr_backend.R removed and tests cleaned up
**Plans:** 2 plans

Plans:
- [x] 03-01-PLAN.md - Investigate Zarr packages (CRAN zarr vs Rarr) with testing and benchmarks
- [x] 03-02-PLAN.md - Document decision and implement chosen path (migrate to CRAN zarr)

### Phase 4: Test Coverage
**Goal**: Package achieves 80%+ overall test coverage with all critical backends covered
**Depends on**: Phase 3
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07
**Success Criteria** (what must be TRUE):
  1. zarr_backend.R coverage is 80%+ (or backend removed)
  2. h5_backend.R coverage is 80%+ (or documented S4 limitation)
  3. as_delayed_array.R coverage is 80%+ (or documented S4 limitation)
  4. as_delayed_array_dataset.R coverage is 80%+
  5. dataset_methods.R coverage is 80%+ (or documented limitation)
  6. as_delarr.R coverage is 80%+ (PRIMARY lazy array interface)
  7. Overall package test coverage is 80%+ (or documented reason if lower)
**Plans:** 5 plans

Plans:
- [x] 04-01-PLAN.md - Create test helpers and zarr_backend tests
- [x] 04-02-PLAN.md - Create h5_backend tests
- [x] 04-03-PLAN.md - Create as_delayed_array and dataset_methods tests
- [x] 04-04-PLAN.md - Run coverage analysis and document results
- [x] 04-05-PLAN.md - Gap closure: as_delarr.R coverage (67.3% -> 80%+)

### Phase 5: Final Validation
**Goal**: Package passes R CMD check --as-cran with 0 errors and documented warnings/notes
**Depends on**: Phase 4
**Requirements**: CRAN-07
**Success Criteria** (what must be TRUE):
  1. R CMD check --as-cran produces 0 errors
  2. R CMD check --as-cran produces 0 warnings (or only documented acceptable warnings for non-CRAN dependencies)
  3. R CMD check --as-cran produces 0 notes (or only documented acceptable notes)
  4. Package installs cleanly on fresh R session
  5. All examples run without errors
  6. All vignettes build successfully
**Plans:** 3 plans

Plans:
- [x] 05-01-PLAN.md - Fix DESCRIPTION dependencies (remove Remotes, add blosc/DelayedArray/DelayedMatrixStats)
- [x] 05-02-PLAN.md - Fix test files (add blosc skip, remove rhdf5 references)
- [x] 05-03-PLAN.md - Run final R CMD check and create cran-comments.md

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. CRAN Quick Wins | 3/3 | Complete | 2026-01-22 |
| 2. Tech Debt | 2/2 | Complete | 2026-01-22 |
| 3. Zarr Decision | 2/2 | Complete | 2026-01-22 |
| 4. Test Coverage | 5/5 | Complete | 2026-01-22 |
| 5. Final Validation | 3/3 | Complete | 2026-01-22 |

---
*Roadmap created: 2026-01-22*
*Last updated: 2026-01-22 (Phase 5 complete)*
