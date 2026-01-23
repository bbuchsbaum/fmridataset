# Project Milestones: fmridataset

## v0.9.0 CRAN-Ready (Shipped: 2026-01-22)

**Delivered:** Package passes R CMD check with 0 errors and is ready for CRAN submission once upstream dependencies (delarr, bidser, fmristore) are accepted.

**Phases completed:** 1-5 (15 plans total)

**Key accomplishments:**

- R CMD check compliance: 0 errors, documented warnings for non-CRAN dependencies
- Zarr backend migrated from broken Bioconductor Rarr to working CRAN zarr (experimental, v3-only)
- H5 backend resource leaks fixed with on.exit() protection
- Test coverage improved: zarr_backend 5% -> 94.6%, as_delayed_array_dataset -> 100%
- 73.3% overall coverage (1991 tests passing) with documented S4 limitations
- cran-comments.md prepared for CRAN submission

**Stats:**

- 91 files modified
- ~73,000 lines of R code
- 5 phases, 15 plans
- Completed in 1 day (2026-01-22)

**Git range:** `fb42ca4` -> `ebf1379`

**What's next:** CRAN submission after delarr, bidser, fmristore are accepted on CRAN

**Archive:** [v0.9.0-ROADMAP.md](milestones/v0.9.0-ROADMAP.md) | [v0.9.0-REQUIREMENTS.md](milestones/v0.9.0-REQUIREMENTS.md)

---
