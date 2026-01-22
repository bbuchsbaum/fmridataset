# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Backend-agnostic fMRI data access: one API works across all storage formats, with lazy loading, chunked iteration, and multi-subject support.
**Current focus:** Phase 3 - Zarr Decision

## Current Position

Phase: 3 of 5 (Zarr Decision)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-01-22 — Completed 03-01-PLAN.md (Zarr investigation)

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 3.75 min
- Total execution time: 0.375 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. CRAN Quick Wins | 3 | 7.5 min | 2.5 min |
| 2. Tech Debt | 2 | 3 min | 1.5 min |
| 3. Zarr Decision | 1 | 15 min | 15 min |

**Recent Trend:**
- Last 5 plans: 01-03 (2.7min), 02-01 (2min), 02-02 (<1min), 03-01 (15min)
- Trend: Phase 3 investigation took longer due to package testing and benchmarking

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **migrate-zarr: Use CRAN zarr package** - Pure CRAN dependency, accept Zarr v3-only limitation, mark as experimental (03-01)
- **Accept no Zarr v2 support** - Users must work with Zarr v3 stores; no legacy compatibility (03-01)
- Target 80% coverage: Balance thoroughness with pragmatism
- Investigate Zarr fully: User wants cloud-native support if viable
- Fix check issues before adding coverage: Unblocks CI/CD quality gates
- Alphabetize DESCRIPTION Suggests: Improves maintainability with 24+ dependencies (01-01)
- No version constraints on new dependencies: Maximum compatibility unless specific need (01-01)
- Use proper regex syntax in .Rbuildignore: Anchors and escaped dots prevent unintended matches (01-03)
- No Bioconductor dependencies: DelayedArray, DelayedMatrixStats excluded to preserve CRAN eligibility (01-01)
- Prefer hdf5r over rhdf5: hdf5r is the preferred HDF5 library for this package (01-01)
- Use utils::getS3method() for S3 method introspection: Respects namespaces and dispatch rules (02-02)
- Use on.exit(add = TRUE, after = FALSE) for resource cleanup: Ensures cleanup runs in reverse order (02-01)
- Register cleanup handlers immediately after resource creation: Prevents leaks even when errors occur (02-01)

### Pending Todos

None yet.

### Blockers/Concerns

**From Requirements:**
- Cannot submit to CRAN until delarr, bidser, fmristore are accepted (external dependency)

**From Phase 3 Investigation:**
- CRAN zarr is very new (0.1.1, Dec 2025) - may discover bugs during implementation
- Cloud path support unclear - needs testing during implementation
- Zarr v3-only limitation will affect documentation (must warn users about v2 incompatibility)

## Session Continuity

Last session: 2026-01-22 (phase execution)
Stopped at: Completed 03-01-PLAN.md (Zarr investigation complete)
Resume file: None

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-22*
