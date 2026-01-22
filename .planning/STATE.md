# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Backend-agnostic fMRI data access: one API works across all storage formats, with lazy loading, chunked iteration, and multi-subject support.
**Current focus:** Phase 4 - Test Coverage

## Current Position

Phase: 4 of 5 (Test Coverage)
Plan: 4 of 4 in current phase
Status: Phase complete
Last activity: 2026-01-22 — Completed 04-04-PLAN.md (Coverage Analysis)

Progress: [████████░░] 75%

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 6.2 min
- Total execution time: 1.14 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. CRAN Quick Wins | 3 | 7.5 min | 2.5 min |
| 2. Tech Debt | 2 | 3 min | 1.5 min |
| 3. Zarr Decision | 2 | 27 min | 13.5 min |
| 4. Test Coverage | 4 | 31 min | 7.75 min |

**Recent Trend:**
- Last 5 plans: 03-02 (12min), 04-01 (8min), 04-02 (10min), 04-03 (3min), 04-04 (10min)
- Trend: Phase 4 complete - coverage analysis and reporting done

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **Accept 73% overall coverage as pragmatic** - Core functionality well-tested; gap in h5_backend due to S4 mocking limitations (04-04)
- **Document h5_backend testing gap as technical debt** - Cannot mock S4 objects; would need real H5 fixtures (medium investment) (04-04)
- **Use mock objects instead of real H5 files for testing** - Tests h5_backend logic without requiring real fmristore files; faster, no write permissions needed (04-02)
- **Create NeuroVol masks not H5NeuroVol in tests** - h5_backend converts H5NeuroVol to NeuroVol internally; use NeuroVol directly (04-02)
- **Test backend_close behavior as-is** - backend_close doesn't modify caller's object due to R copy-on-write; test actual behavior (04-01)
- **Use helper functions for backend tests** - create_test_zarr() provides consistent test data with reproducible seeding (04-01)
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

**From Phase 4 (Complete):**
- ✅ Overall package coverage: 73% (target: 80%, gap: -7%)
- ✅ zarr_backend coverage: 94.6% (exceeds 80% target) - 04-01
- ✅ as_delayed_array_dataset coverage: 100% (perfect) - 04-03
- ⚠️ h5_backend coverage: 30.1% (S4 mocking limitation) - 04-04
- ℹ️ Coverage gap concentrated in optional features (h5_backend)
- ℹ️ 8 h5_backend tests skipped - require real S4 objects
- 📄 COVERAGE-REPORT.md documents gaps and recommendations

**Technical Debt:**
- h5_backend testing gap (30.1% coverage) - would need real H5 fixtures (medium investment)

**For Next Phase:**
- Need to update README with Zarr backend documentation
- Consider adding Zarr vignette section if useful
- Should test cloud storage paths when accessible

## Session Continuity

Last session: 2026-01-22 (phase execution)
Stopped at: Completed 04-04-PLAN.md (Phase 4 complete)
Resume file: None

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-22*
