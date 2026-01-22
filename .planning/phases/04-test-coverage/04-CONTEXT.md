# Phase 4: Test Coverage - Context

**Gathered:** 2026-01-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Achieve 80%+ test coverage across zarr_backend, h5_backend, as_delayed_array, as_delayed_array_dataset, and dataset_methods. Write tests for existing functionality — no new features, just validation of current code paths.

</domain>

<decisions>
## Implementation Decisions

### Dependency Handling
- Use `skip_if_not_installed()` for optional backends (hdf5r, zarr)
- Tests skip gracefully on systems without deps — no failures, no broken CI
- If hdf5r is installed but broken (load errors), let tests fail — that's a real problem to surface
- Zarr backend gets same rigor as other backends despite EXPERIMENTAL status
- Cloud storage paths (S3/GCS/Azure) are out of scope — test local Zarr stores only

### Test Data Strategy
- Generate test data on-the-fly in test setup, clean up in teardown
- No committed fixture files — keeps repo lean
- Use minimal dimensions: ~4x4x4 volume, ~10 timepoints — fast tests that cover logic
- Create shared helpers in `tests/testthat/helper-backends.R`:
  - `create_test_h5()` — generates temp HDF5 file
  - `create_test_zarr()` — generates temp Zarr store
  - Common synthetic data patterns
- Use `withr::local_tempdir()` for automatic cleanup — files always removed

### Claude's Discretion
- Exact coverage strategy to hit 80% target
- Which edge cases and error paths to prioritize
- Test organization within files
- Whether to add integration tests beyond unit tests

</decisions>

<specifics>
## Specific Ideas

- Zarr tests should work despite EXPERIMENTAL status — same coverage bar
- Keep test runtime fast with tiny data dimensions
- Follow existing testthat patterns in the package

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-test-coverage*
*Context gathered: 2026-01-22*
