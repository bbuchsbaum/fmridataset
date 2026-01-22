---
phase: 03-zarr-decision
plan: 01
subsystem: storage-backend
tags: [zarr, cran, bioconductor, performance, cloud-storage]

# Dependency graph
requires:
  - phase: 02-tech-debt
    provides: Clean H5 resource management for comparison baseline
provides:
  - Zarr package investigation comparing CRAN zarr vs Rarr
  - Performance benchmarks (Zarr vs HDF5)
  - Decision to migrate to CRAN zarr package
affects: [03-02-implementation]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - .planning/phases/03-zarr-decision/03-INVESTIGATION.md
  modified: []

key-decisions:
  - "migrate-zarr: Use CRAN zarr package despite Bioconductor Rarr being production-ready"
  - "Accept Zarr v3-only limitation (no legacy v2 support)"
  - "Mark zarr backend as experimental"

patterns-established: []

# Metrics
duration: 15min
completed: 2026-01-22
---

# Phase 3 Plan 01: Zarr Backend Investigation Summary

**CRAN zarr package evaluated; decision to migrate from Rarr despite v2 incompatibility and maturity concerns**

## Performance

- **Duration:** 15 min
- **Started:** 2026-01-22T~14:30:00Z (estimated)
- **Completed:** 2026-01-22T~14:45:00Z (estimated)
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Tested CRAN zarr package core operations (8/8 tests pass)
- Tested Rarr package and benchmarked against HDF5 (Rarr 6.4x faster for timepoint access)
- Documented decision to migrate to CRAN zarr despite trade-offs

## Task Commits

Each task was committed atomically:

1. **Task 1: Test CRAN zarr package core operations** - `d7c6ca7` (docs)
2. **Task 2: Test Rarr and benchmark vs HDF5** - `7210e4d` (docs)
3. **Task 3: Record decision checkpoint outcome** - (this commit)

**Plan metadata:** (this commit - docs: complete Zarr investigation plan)

## Files Created/Modified

- `.planning/phases/03-zarr-decision/03-INVESTIGATION.md` - Investigation findings, performance data, and decision documentation

## Decisions Made

**Decision: migrate-zarr (use CRAN zarr package)**

Investigation revealed Rarr is production-ready with Zarr v2 support, cloud capabilities, and superior performance for fMRI workloads (6.4x faster single timepoint access). However, user chose CRAN zarr because:

1. **Pure CRAN dependency** - Eliminates Bioconductor dependency chain
2. **Zarr v2 not needed** - Package won't support legacy files anyway
3. **Current implementation broken** - Existing zarr_backend.R uses wrong API calls
4. **Ecosystem improvement** - Stress test new CRAN package, report bugs, help mature the ecosystem

**Trade-offs accepted:**
- No Zarr v2 compatibility (cannot read most existing public datasets)
- Very new package (0.1.1, Dec 2025 - only 1 month old)
- Cloud support undocumented (may work, but untested)
- Requires zarr backend marked as experimental

## Deviations from Plan

None - plan executed exactly as written.

Investigation was a checkpoint-based plan. User made informed decision based on presented evidence.

## Issues Encountered

None - all tests passed cleanly for both packages.

## User Setup Required

None - no external service configuration required.

This was an investigation plan producing documentation, not code changes.

## Next Phase Readiness

**Ready for 03-02 implementation:**
- Decision documented with clear rationale
- Investigation findings available for reference
- Package APIs understood (both CRAN zarr R6 and current Rarr)
- Performance baseline established

**Blockers/concerns:**
- CRAN zarr is very new (0.1.1, Dec 2025) - may discover bugs during implementation
- Cloud path support unclear - needs testing during implementation
- Zarr v3-only limitation will affect documentation (must warn users about v2 incompatibility)

**Implementation notes for 03-02:**
- Mark zarr backend as experimental in docs
- Test local filesystem access first, then cloud paths if possible
- Plan for potential bug reports to CRAN zarr maintainer
- Update all tests to use Zarr v3 stores (not v2)

---
*Phase: 03-zarr-decision*
*Completed: 2026-01-22*
