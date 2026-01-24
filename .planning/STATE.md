# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-23)

**Core value:** Backend-agnostic fMRI data access: one API works across all storage formats, with lazy loading, chunked iteration, and multi-subject support.
**Current focus:** v0.9.1 Documentation Quality

## Current Position

Phase: 7 - Developer Vignettes (In Progress)
Plan: 02 of 4
Status: In progress
Last activity: 2026-01-24 — Completed 07-02-PLAN.md

Progress: [====================>------------------] 1.2/3 phases

## Milestone Overview

**Milestone:** v0.9.1 Documentation Quality
**Goal:** Vignettes with executable examples, accurate API, clear explanations; polished pkgdown site.
**Phases:** 3 (6, 7, 8)
**Requirements:** 29 total

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 6 | User Vignettes | 12 | ✓ Complete |
| 7 | Developer Vignettes | 9 | In Progress (2/4 plans) |
| 8 | Documentation Infrastructure | 8 | Pending |

## Shipped Milestones

- **v0.9.0 CRAN-Ready** (2026-01-22)
  - 5 phases, 15 plans
  - R CMD check: 0 errors
  - 73.3% test coverage
  - Archive: `.planning/milestones/v0.9.0-*`

## Accumulated Context

### Decisions

Key decisions from v0.9.0 milestone are documented in:
- `.planning/PROJECT.md` (Key Decisions table)
- `.planning/milestones/v0.9.0-ROADMAP.md` (full archive)

**v0.9.1 Decisions:**

| Date | Decision | Rationale | Phase |
|------|----------|-----------|-------|
| 2026-01-23 | Use foreach loops instead of for loops for chunk iterators | R iterators don't work with for loops (return environment not values) | 06-01 |
| 2026-01-23 | Use blockids() instead of samples() for run boundaries | samples() returns time points in seconds, not run-specific indices | 06-01 |
| 2026-01-23 | Reorganize See Also section with sequential reading order | Guide users through clear learning path: intro → architecture → h5 → study | 06-01 |
| 2026-01-23 | Use consistent voxel counts across subjects in vignette examples | Real studies have common registration space, inconsistent dims cause backend validation failure | 06-04 |
| 2026-01-23 | Fix TR comparison with unname() in study dataset creation | Named vectors cause all.equal() to return string mismatch instead of TRUE | 06-04 |
| 2026-01-23 | Use nextElem() iterator pattern for chunk processing | For loops don't work with iterator objects, need explicit iteration protocol | 06-04 |
| 2026-01-24 | Use demo_ prefix for vignette backends to avoid collisions | Clear distinction between demo code and production backends | 07-02 |
| 2026-01-24 | Simulate file operations in demo backends for executable examples | Enables vignette rendering without real files or file system dependencies | 07-02 |
| 2026-01-24 | Core backend implementation executes, advanced patterns are demonstrations | Core neurostream example is self-contained and reproducible, complex integrations require additional infrastructure | 07-03 |
| 2026-01-24 | Use test:// protocol instead of neurostream:// for executable example | Avoid confusion about requiring actual NeuroStream server | 07-03 |
| 2026-01-24 | Use tempdir() for cache directory | Portable across systems, avoids permission issues | 07-03 |

### Pending Todos

None.

### Blockers/Concerns

**External:**
- Cannot submit to CRAN until delarr, bidser, fmristore are accepted (external dependency)

**Technical Debt (from v0.9.0):**
- h5_backend coverage at 30.1% (S4 mocking limitation)
- TODO in zarr_backend.R:268 - "Optimize for sparse access patterns"

## Session Continuity

Last session: 2026-01-24 05:10 UTC
Stopped at: Completed 07-02-PLAN.md
Resume file: None
Status: Continue with 07-03-PLAN.md or verify phase

---
*State initialized: 2026-01-22*
*Last updated: 2026-01-24 — Phase 7 in progress (07-02 complete)*
