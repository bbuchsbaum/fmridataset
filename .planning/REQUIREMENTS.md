# Requirements: fmridataset CRAN-Ready

**Defined:** 2026-01-22
**Core Value:** Backend-agnostic fMRI data access that passes CRAN standards

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### CRAN Compliance

- [x] **CRAN-01**: Fix unstated test dependencies (add devtools, iterators, withr to Suggests - Bioconductor deps excluded)
- [x] **CRAN-02**: Fix undefined global function `generate_all_golden_data`
- [x] **CRAN-03**: Fix Rd cross-reference for `sampling_frame` link (add package anchor)
- [x] **CRAN-04**: Add missing vignette dependencies (microbenchmark, pryr) to Suggests
- [x] **CRAN-05**: Add `.planning` to .Rbuildignore
- [x] **CRAN-06**: Handle non-standard top-level files (move or add to .Rbuildignore)
- [ ] **CRAN-07**: Pass `R CMD check --as-cran` with 0 errors, 0 warnings, 0 notes

### Test Coverage

- [ ] **TEST-01**: Increase zarr_backend.R coverage from 5% to 80%+
- [ ] **TEST-02**: Increase h5_backend.R coverage from 26% to 80%+
- [ ] **TEST-03**: Increase as_delayed_array.R coverage from 10% to 80%+
- [ ] **TEST-04**: Increase as_delayed_array_dataset.R coverage from 6% to 80%+
- [ ] **TEST-05**: Increase dataset_methods.R coverage from 24% to 80%+
- [ ] **TEST-06**: Achieve 80%+ overall package test coverage

### Zarr Investigation

- [ ] **ZARR-01**: Assess Rarr package viability (stability, CRAN compatibility, maintenance)
- [ ] **ZARR-02**: Test Zarr cloud path support (S3, GCS, Azure URLs)
- [ ] **ZARR-03**: Make go/no-go decision on Zarr support (keep as Suggests or remove)

### Tech Debt

- [ ] **DEBT-01**: Fix H5 backend resource leak in metadata retrieval (on.exit cleanup)
- [ ] **DEBT-02**: Fix H5 backend resource leak in data reading (on.exit cleanup)
- [ ] **DEBT-03**: Commit pending storage_backend.R fix (getS3method change)

## v2 Requirements

Deferred to future release. Not in current roadmap.

### Performance

- **PERF-01**: Cache H5 dimension queries to avoid repeated file opens
- **PERF-02**: Cache NIfTI metadata at backend initialization

### Features

- **FEAT-01**: Streaming I/O for Zarr backend
- **FEAT-02**: Sparse matrix support for backends

## Out of Scope

Explicitly excluded from this milestone.

| Feature | Reason |
|---------|--------|
| Actual CRAN submission | Wait for delarr, bidser, fmristore to be on CRAN first |
| New feature development | Focus is quality, not features |
| Major API changes | Backward compatibility required |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CRAN-01 | Phase 1 | Complete |
| CRAN-02 | Phase 1 | Complete |
| CRAN-03 | Phase 1 | Complete |
| CRAN-04 | Phase 1 | Complete |
| CRAN-05 | Phase 1 | Complete |
| CRAN-06 | Phase 1 | Complete |
| DEBT-01 | Phase 2 | Pending |
| DEBT-02 | Phase 2 | Pending |
| DEBT-03 | Phase 2 | Pending |
| ZARR-01 | Phase 3 | Pending |
| ZARR-02 | Phase 3 | Pending |
| ZARR-03 | Phase 3 | Pending |
| TEST-01 | Phase 4 | Pending |
| TEST-02 | Phase 4 | Pending |
| TEST-03 | Phase 4 | Pending |
| TEST-04 | Phase 4 | Pending |
| TEST-05 | Phase 4 | Pending |
| TEST-06 | Phase 4 | Pending |
| CRAN-07 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0

---
*Requirements defined: 2026-01-22*
*Last updated: 2026-01-22 after roadmap creation*
