# Plan 01-02 Summary: Fix golden_data and sampling_frame cross-reference

## Status: Complete

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Delete golden data generation file | 693301b | R/golden_data_generation.R (deleted) |
| 2 | Fix sampling_frame cross-reference | d5ae285 | R/all_generic.R, man/get_TR.Rd |

## What Was Built

- **Removed misplaced test utilities**: Deleted R/golden_data_generation.R which contained test data generation functions that shouldn't be in package exports. The functionality exists appropriately in tests/testthat/helper-golden.R.

- **Fixed cross-package documentation**: Updated get_TR documentation to use `[fmrihrf::sampling_frame()]` syntax instead of `\code{\link{sampling_frame}}`, providing proper package-anchored cross-reference.

## Deliverables

- [x] R/golden_data_generation.R deleted
- [x] NAMESPACE no longer exports generate_golden_test_data, update_golden_test_data, validate_golden_test_data
- [x] R/all_generic.R uses fmrihrf::sampling_frame() cross-reference
- [x] man/get_TR.Rd regenerated with proper package anchor

## Deviations

None.

## Issues Encountered

None.

---
*Completed: 2026-01-22*
