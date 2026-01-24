# Roadmap: fmridataset v0.9.1

## Overview

**Milestone:** v0.9.1 Documentation Quality
**Goal:** Ensure all 7 vignettes have executable examples, accurate API usage, and clear explanations; rebuild pkgdown site with polished documentation.
**Phases:** 3 (continuing from v0.9.0, starting at Phase 6)
**Requirements:** 29 total

## Phase Summary

| # | Name | Goal | Requirements | Criteria |
|---|------|------|--------------|----------|
| 6 | User Vignettes | Users can learn fmridataset from executable, accurate documentation | 12 (VIG-01 to VIG-06, VIG-16 to VIG-21) | 4 |
| 7 | Developer Vignettes | Developers can extend fmridataset using accurate, runnable guides | 9 (VIG-07 to VIG-15) | 4 |
| 8 | Documentation Infrastructure | Package documentation builds cleanly and deploys correctly | 8 (PKG-01 to PKG-05, ROX-01 to ROX-03) | 5 |

## Phases

### Phase 6: User Vignettes

**Goal:** Users can learn fmridataset from executable, accurate documentation covering getting started, architecture, HDF5 usage, and multi-subject analysis.

**Requirements covered:**
- VIG-01: fmridataset-intro.Rmd has executable examples
- VIG-02: fmridataset-intro.Rmd content matches current API
- VIG-03: fmridataset-intro.Rmd has clear user-centric explanations
- VIG-04: architecture-overview.Rmd has executable examples
- VIG-05: architecture-overview.Rmd content matches current API
- VIG-06: architecture-overview.Rmd has clear explanations
- VIG-16: h5-backend-usage.Rmd has executable examples
- VIG-17: h5-backend-usage.Rmd content matches current API
- VIG-18: h5-backend-usage.Rmd has clear explanations
- VIG-19: study-level-analysis.Rmd has executable examples
- VIG-20: study-level-analysis.Rmd content matches current API
- VIG-21: study-level-analysis.Rmd has clear explanations

**Success criteria:**
1. User can render fmridataset-intro.Rmd without errors (examples execute)
2. User can render architecture-overview.Rmd without errors (examples execute)
3. User can render h5-backend-usage.Rmd without errors (examples execute)
4. User can render study-level-analysis.Rmd without errors (examples execute)

**Depends on:** None

**Plans:** 4 plans
Plans:
- [ ] 06-01-PLAN.md — Fix fmridataset-intro.Rmd executable examples
- [ ] 06-02-PLAN.md — Fix architecture-overview.Rmd executable examples
- [ ] 06-03-PLAN.md — Fix h5-backend-usage.Rmd executable examples
- [ ] 06-04-PLAN.md — Fix study-level-analysis.Rmd executable examples

---

### Phase 7: Developer Vignettes

**Goal:** Developers can extend fmridataset using accurate, runnable guides for backend creation, registry usage, and advanced customization.

**Requirements covered:**
- VIG-07: backend-development-basics.Rmd has executable examples
- VIG-08: backend-development-basics.Rmd content matches current API
- VIG-09: backend-development-basics.Rmd has clear explanations
- VIG-10: backend-registry.Rmd has executable examples
- VIG-11: backend-registry.Rmd content matches current API
- VIG-12: backend-registry.Rmd has clear explanations
- VIG-13: extending-backends.Rmd has executable examples
- VIG-14: extending-backends.Rmd content matches current API
- VIG-15: extending-backends.Rmd has clear explanations

**Success criteria:**
1. Developer can render backend-development-basics.Rmd without errors (examples execute)
2. Developer can render backend-registry.Rmd without errors (examples execute)
3. Developer can render extending-backends.Rmd without errors (examples execute)
4. Developer following any backend vignette can create a minimal working backend

**Depends on:** None (can run parallel with Phase 6)

---

### Phase 8: Documentation Infrastructure

**Goal:** Package documentation builds cleanly, reference docs are complete, and pkgdown site is ready for deployment.

**Requirements covered:**
- ROX-01: All exported functions have complete documentation
- ROX-02: Function examples execute without errors
- ROX-03: Documentation builds without warnings
- PKG-01: _pkgdown.yml has complete site configuration
- PKG-02: pkgdown site builds without errors
- PKG-03: Reference documentation renders correctly
- PKG-04: Vignette articles render correctly
- PKG-05: Site is deployed/ready for deployment

**Success criteria:**
1. `devtools::document()` completes with 0 warnings
2. `devtools::run_examples()` completes with 0 errors
3. `pkgdown::build_site()` completes with 0 errors
4. All 7 vignettes appear in pkgdown articles section
5. All exported functions appear in pkgdown reference section with working examples

**Depends on:** Phase 6, Phase 7 (vignettes must be fixed before site build)

---

## Coverage Validation

### By Requirement

| Requirement | Phase | Description |
|-------------|-------|-------------|
| VIG-01 | 6 | fmridataset-intro.Rmd executable examples |
| VIG-02 | 6 | fmridataset-intro.Rmd API accuracy |
| VIG-03 | 6 | fmridataset-intro.Rmd clear explanations |
| VIG-04 | 6 | architecture-overview.Rmd executable examples |
| VIG-05 | 6 | architecture-overview.Rmd API accuracy |
| VIG-06 | 6 | architecture-overview.Rmd clear explanations |
| VIG-07 | 7 | backend-development-basics.Rmd executable examples |
| VIG-08 | 7 | backend-development-basics.Rmd API accuracy |
| VIG-09 | 7 | backend-development-basics.Rmd clear explanations |
| VIG-10 | 7 | backend-registry.Rmd executable examples |
| VIG-11 | 7 | backend-registry.Rmd API accuracy |
| VIG-12 | 7 | backend-registry.Rmd clear explanations |
| VIG-13 | 7 | extending-backends.Rmd executable examples |
| VIG-14 | 7 | extending-backends.Rmd API accuracy |
| VIG-15 | 7 | extending-backends.Rmd clear explanations |
| VIG-16 | 6 | h5-backend-usage.Rmd executable examples |
| VIG-17 | 6 | h5-backend-usage.Rmd API accuracy |
| VIG-18 | 6 | h5-backend-usage.Rmd clear explanations |
| VIG-19 | 6 | study-level-analysis.Rmd executable examples |
| VIG-20 | 6 | study-level-analysis.Rmd API accuracy |
| VIG-21 | 6 | study-level-analysis.Rmd clear explanations |
| PKG-01 | 8 | _pkgdown.yml complete configuration |
| PKG-02 | 8 | pkgdown site builds without errors |
| PKG-03 | 8 | Reference docs render correctly |
| PKG-04 | 8 | Vignette articles render correctly |
| PKG-05 | 8 | Site ready for deployment |
| ROX-01 | 8 | All exports have complete docs |
| ROX-02 | 8 | Function examples execute |
| ROX-03 | 8 | Docs build without warnings |

### Summary

- Total requirements: 29
- Mapped: 29
- Unmapped: 0

**Coverage by phase:**
- Phase 6 (User Vignettes): 12 requirements
- Phase 7 (Developer Vignettes): 9 requirements
- Phase 8 (Documentation Infrastructure): 8 requirements

---

## Progress

| Phase | Status | Plans | Completed |
|-------|--------|-------|-----------|
| 6 - User Vignettes | Planned | 4 | - |
| 7 - Developer Vignettes | Pending | 0/? | - |
| 8 - Documentation Infrastructure | Pending | 0/? | - |

---
*Created: 2026-01-23*
*Milestone continues from v0.9.0 (phases 1-5)*
