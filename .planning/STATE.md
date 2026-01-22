# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Backend-agnostic fMRI data access: one API works across all storage formats, with lazy loading, chunked iteration, and multi-subject support.
**Current focus:** Phase 4 - Test Coverage

## Current Position

Phase: 4 of 5 (Test Coverage)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-01-22 — Phase 3 complete, verified

Progress: [██████░░░░] 60%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 5.14 min
- Total execution time: 0.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. CRAN Quick Wins | 3 | 7.5 min | 2.5 min |
| 2. Tech Debt | 2 | 3 min | 1.5 min |
| 3. Zarr Decision | 2 | 27 min | 13.5 min |

**Recent Trend:**
- Last 5 plans: 02-01 (2min), 02-02 (<1min), 03-01 (15min), 03-02 (12min)
- Trend: Phase 3 required investigation + implementation, longer than quick fixes

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **Auto-open backends for validation** - backend_get_dims/get_mask auto-open if needed; follows nifti pattern (03-02)
- **Single-array Zarr stores** - Simplified API: no data_key/mask_key parameters; root arrays only (03-02)
- **Mark Zarr as EXPERIMENTAL** - Package is new (0.1.1), needs field testing (03-02)
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

**From Phase 3 Completed:**
- ✅ CRAN zarr integration complete, marked EXPERIMENTAL
- ⚠️ Cloud path support (S3/GCS/Azure) untested - needs verification when accessible
- ⚠️ Zarr v3-only documented - users with v2 stores must convert externally

**For Next Phase:**
- Need to update README with Zarr backend documentation
- Consider adding Zarr vignette section if useful
- Should test cloud storage paths when accessible

## Session Continuity

Last session: 2026-01-22 (phase execution)
Stopped at: Phase 3 complete, ready for Phase 4 planning
Resume file: None

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-22*
