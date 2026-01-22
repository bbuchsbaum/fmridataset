# Requirements: fmridataset CRAN-Ready

**Defined:** 2026-01-22
**Core Value:** Backend-agnostic fMRI data access that passes CRAN standards

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### CRAN Compliance

- [ ] **CRAN-01**: Fix unstated test dependencies (add DelayedArray, rhdf5, devtools, iterators, withr to Suggests)
- [ ] **CRAN-02**: Fix undefined global function `generate_all_golden_data`
- [ ] **CRAN-03**: Fix Rd cross-reference for `sampling_frame` link (add package anchor)
- [ ] **CRAN-04**: Add missing vignette dependencies (microbenchmark, pryr) to Suggests
- [ ] **CRAN-05**: Add `.planning` to .Rbuildignore
- [ ] **CRAN-06**: Handle non-standard top-level files (move or add to .Rbuildignore)
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
| CRAN-01 | TBD | Pending |
| CRAN-02 | TBD | Pending |
| CRAN-03 | TBD | Pending |
| CRAN-04 | TBD | Pending |
| CRAN-05 | TBD | Pending |
| CRAN-06 | TBD | Pending |
| CRAN-07 | TBD | Pending |
| TEST-01 | TBD | Pending |
| TEST-02 | TBD | Pending |
| TEST-03 | TBD | Pending |
| TEST-04 | TBD | Pending |
| TEST-05 | TBD | Pending |
| TEST-06 | TBD | Pending |
| ZARR-01 | TBD | Pending |
| ZARR-02 | TBD | Pending |
| ZARR-03 | TBD | Pending |
| DEBT-01 | TBD | Pending |
| DEBT-02 | TBD | Pending |
| DEBT-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 0
- Unmapped: 19 ⚠️

---
*Requirements defined: 2026-01-22*
*Last updated: 2026-01-22 after initial definition*
