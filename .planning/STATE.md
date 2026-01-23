# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Backend-agnostic fMRI data access: one API works across all storage formats, with lazy loading, chunked iteration, and multi-subject support.
**Current focus:** Milestone complete - CRAN-Ready

## Current Position

Phase: 5 of 5 (Final Validation)
Plan: 3 of 3 in current phase
Status: Milestone complete
Last activity: 2026-01-23 — Completed 05-03-PLAN.md (Final CRAN Validation)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Average duration: 5.6 min
- Total execution time: 1.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. CRAN Quick Wins | 3 | 7.5 min | 2.5 min |
| 2. Tech Debt | 2 | 3 min | 1.5 min |
| 3. Zarr Decision | 2 | 27 min | 13.5 min |
| 4. Test Coverage | 5 | 37 min | 7.4 min |
| 5. Final Validation | 3 | 12 min | 4 min |

**Recent Trend:**
- Last 5 plans: 04-04 (10min), 04-05 (6min), 05-01 (1min), 05-02 (2min), 05-03 (9min)
- Trend: All phases complete - package ready for CRAN

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **Fix testthat::test_dir() call in test_optional_packages.R** - testthat::test_dir() fails during R CMD check; replaced with package availability summary (05-03)
- **Add blosc skip for all zarr::as_zarr() calls** - zarr compression requires blosc codec; tests must skip when unavailable (05-02)
- **Eliminate rhdf5 (Bioconductor) in favor of hdf5r (CRAN)** - Package policy is CRAN-only where possible; hdf5r is production HDF5 library (05-02)
- **Add blosc, DelayedArray, DelayedMatrixStats to Suggests for test support** - blosc required by zarr compression, DelayedArray/DelayedMatrixStats used in tests (05-01)
- **Remove Remotes field for CRAN compliance** - Remotes not recognized by CRAN; GitHub deps need alternative strategy (05-01)
- **Test error paths via backend_get_data for delarr** - delarr pre-validates indices, so pull_fun error branches unreachable; test underlying function directly (04-05)
- **Skip nifti_backend tests** - S4 dependency like h5_backend; focus on matrix/study backends (04-05)
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
- ✅ Overall package coverage: 73.3% (target: 80%, gap: -6.7%)
- ✅ zarr_backend coverage: 94.6% (exceeds 80% target) - 04-01
- ✅ as_delayed_array_dataset coverage: 100% (perfect) - 04-03
- ✅ as_delarr.R coverage: 81.5% (up from 67.3%) - 04-05 gap closure
- ⚠️ h5_backend coverage: 30.1% (S4 mocking limitation) - 04-04
- ℹ️ Coverage gap concentrated in S4-dependent backends (h5, nifti)
- ℹ️ 8 h5_backend tests skipped - require real S4 objects
- 📄 COVERAGE-REPORT.md documents gaps and recommendations

**Technical Debt:**
- h5_backend testing gap (30.1% coverage) - would need real H5 fixtures (medium investment)

**From Phase 5 (Complete):**
- ✅ DESCRIPTION CRAN-compatible (Remotes removed, test deps added) - 05-01
- ✅ Test dependencies properly guarded (blosc, hdf5r skips added) - 05-02
- ✅ No rhdf5/Bioconductor references remain in tests - 05-02
- ✅ R CMD check --as-cran passes with 0 errors - 05-03
- ✅ cran-comments.md documents check results - 05-03
- ⚠️ GitHub-only dependencies still in Imports/Suggests (delarr, bidser, fmristore)

**Package Status:**
- R CMD check: 0 errors, 1 warning (non-CRAN deps), 1 note (new submission)
- All check results documented in cran-comments.md
- Package ready for CRAN submission after dependencies accepted
- Optional enhancements: Update README with Zarr documentation, add Zarr vignette

## Session Continuity

Last session: 2026-01-23 (phase execution)
Stopped at: Completed 05-03-PLAN.md (final CRAN validation)
Resume file: None
Status: All phases complete

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-23*
*Project complete: All 5 phases executed successfully*
