# Testing Patterns

**Analysis Date:** 2026-01-22

## Test Framework

**Runner:**
- testthat (v3.0.0+)
- Edition 3 (configured in `tests/testthat/_testthat.yml`)
- Config file: `tests/testthat/_testthat.yml` (minimal, just edition setting)

**Assertion Library:**
- `testthat` built-in expectations: `expect_equal()`, `expect_true()`, `expect_s3_class()`, etc.

**Run Commands:**
```bash
# Run all tests
devtools::test()
testthat::test_dir("tests/testthat")

# Run a single test file
testthat::test_file("tests/testthat/test_sampling_frame.R")

# Run tests matching a pattern
devtools::test(filter = "backend")

# Run with coverage
covr::package_coverage()
```

## Test File Organization

**Location:**
- All tests in `tests/testthat/`
- Pattern: `test_[component].R` or `test-[type].R`
- Total: 40+ test files, ~11,000 lines of test code

**Naming:**
Test files use consistent pattern:
- `test_sampling_frame.R` - sampling frame functionality
- `test_dataset.R` - dataset constructors
- `test_integration.R` - cross-component workflows
- `test_*_backend.R` - backend-specific tests (e.g., `test_zarr_backend.R`, `test_matrix_backend.R`)
- `test_data_chunks*.R` - chunking strategies (multiple files for comprehensive coverage)
- `test_fmri_series_*.R` - FmriSeries class tests (multiple files)
- `test-golden-*.R` - Golden/snapshot tests (e.g., `test-golden-datasets.R`, `test-golden-snapshots.R`)
- `test_error_*.R` - Error handling (e.g., `test_error_robustness.R`, `test_error_constructors.R`)
- Special cases: `test_memory_safety.R`, `test_cache_management.R`, `test_dummy_mode.R`

**Coverage Areas:**
- All backend implementations (matrix, nifti, h5, zarr, study)
- Data access patterns and chunking strategies
- Type conversions and metadata handling
- Edge cases and error conditions
- Performance optimizations (e.g., NIfTI caching)
- Backward compatibility
- Memory safety and resource cleanup

## Test Structure

**Suite Organization:**
All tests use `test_that()` blocks:
```r
library(fmridataset)

test_that("sampling_frame utilities work", {
  sf <- fmrihrf::sampling_frame(blocklens = c(10, 20, 30), TR = 2)

  expect_true(is.sampling_frame(sf))
  expect_equal(get_TR(sf), 2)
  expect_equal(n_runs(sf), 3)
})
```

**Patterns:**

*Setup pattern:*
```r
test_that("complete workflow with matrix backend", {
  # Setup: Create test data
  n_timepoints <- 100
  n_voxels <- 50
  set.seed(123)
  time_series <- matrix(0, nrow = n_timepoints, ncol = n_voxels)

  # Create backend
  backend <- matrix_backend(data_matrix = time_series)
```

*Teardown pattern:*
```r
  # Cleanup explicit in with_mocked_bindings context
  unlink(c(temp_files, temp_mask))
})
```

*Context setup (where needed):*
```r
# Load reference data (golden tests)
ref_data <- load_golden_data("reference_data")

# Create mock NIfTI files
for (f in c(temp_files, temp_mask)) {
  file.create(f)
}
```

*Assertion pattern:*
```r
expect_equal(get_TR(dataset$sampling_frame), 2)
expect_equal(n_runs(dataset$sampling_frame), 2)
expect_s3_class(dset, "fmri_dataset")
expect_true(is.logical(mask))
expect_length(mask, n_voxels)
```

## Mocking

**Framework:** `mockery` and `mockr` packages

**Pattern from test_dataset.R:**
```r
with_mocked_bindings(
  nifti_backend = function(source, mask_source, preload = FALSE, ...) {
    # Create mock backend bypassing validation
    backend <- matrix_backend(matrix(rnorm(1000), 100, 10))
    class(backend) <- c("nifti_backend", "storage_backend")
    backend
  },
  backend_get_dims.nifti_backend = function(backend) {
    list(spatial = c(10, 1, 1), time = 300)
  },
  backend_get_mask.nifti_backend = function(backend) {
    rep(TRUE, 10)
  },
  validate_backend = function(backend) TRUE,
  .package = "fmridataset",
  {
    dset <- fmri_dataset(
      scans = temp_files,
      mask = temp_mask,
      run_length = c(100, 100, 100),
      TR = 2
    )
    expect_true(!is.null(dset))
  }
)
```

**What to Mock:**
- File I/O functions when testing without actual files
- Backend validation functions to skip expensive checks
- System-dependent functions (file.exists, file.create)

**What NOT to Mock:**
- Core data structures (matrix_backend, data_chunk)
- Utility functions (blocklens, get_TR)
- Public API functions being tested

## Fixtures and Factories

**Test Data:**
Test data created inline in most tests using:
- `matrix(rnorm(...), nrow = X, ncol = Y)` for synthetic matrices
- `set.seed(123)` for reproducibility

**Golden/Reference Data:**
Loaded from cached golden data objects:
```r
ref_data <- load_golden_data("reference_data")
ref_data <- load_golden_data("matrix_dataset")
```

**Location:**
- Helper functions in test files as needed
- No central fixture library; each test creates what it needs
- Golden data loaded via `load_golden_data()` helper function (test-support utility)

**Factory Pattern:**
Used for complex objects:
```r
# Create NeuroVec objects for testing
scans <- lapply(1:length(unique(facedes$run)), function(i) {
  arr <- array(rnorm(10 * 10 * 10 * 244), c(10, 10, 10, 244))
  bspace <- neuroim2::NeuroSpace(dim = c(10, 10, 10, 244))
  neuroim2::NeuroVec(arr, bspace)
})

mask <- neuroim2::LogicalNeuroVol(
  array(rnorm(10 * 10 * 10), c(10, 10, 10)) > 0,
  neuroim2::NeuroSpace(dim = c(10, 10, 10))
)
```

## Coverage

**Requirements:**
- Codecov integration configured in `codecov.yml`
- Patch coverage target: 70%
- Project coverage: auto (informational)
- Coverage reports: precision 2, rounded down, range 70-100%

**View Coverage:**
```bash
# Generate coverage report
covr::package_coverage()

# View in browser
browseURL(file.path(getwd(), "coverage.html"))
```

**GitHub Actions:**
- Test coverage workflow: `.github/workflows/test-coverage.yaml`
- Automatic coverage reporting on PRs

## Test Types

**Unit Tests:**
- Scope: Individual functions and methods
- Approach: Direct function calls with controlled inputs
- Examples: `test_sampling_frame.R` (sampling_frame utilities), `test_integration.R` (basic backend operations)
- Assertion style: `expect_equal()`, `expect_true()`, `expect_s3_class()`

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Create datasets → access data → manipulate → verify end-to-end
- Examples: `test_integration.R` (complete workflow with matrix backend), `test_study_integration.R`
- Pattern: Setup data → create dataset → execute operations → verify results

```r
test_that("complete workflow with matrix backend", {
  # 1. Create data
  time_series <- matrix(0, nrow = 100, ncol = 50)
  # 2. Create backend
  backend <- matrix_backend(data_matrix = time_series)
  # 3. Create dataset
  dataset <- fmri_dataset(scans = backend, TR = 2, run_length = c(50, 50))
  # 4-8. Test full pipeline
  expect_equal(get_TR(dataset$sampling_frame), 2)
  # ...
})
```

**E2E Tests:**
- Not formally structured; integration tests serve this purpose
- Golden/snapshot tests provide E2E verification: `test-golden-*.R`
- Verify consistency across versions and configurations

**Snapshot/Golden Tests:**
- Files: `test-golden-*.R` (datasets, backends, snapshots, sampling-frame)
- Pattern: Load reference data, create output, compare to golden
- Used for: Output consistency, regression detection, format validation

```r
test_that("dataset print output matches snapshot", {
  ref_data <- load_golden_data("reference_data")
  dset <- matrix_dataset(ref_data$matrix_data, TR = ref_data$metadata$TR, ...)

  expect_snapshot({
    print(dset)
  })
})
```

## Common Patterns

**Async Testing:**
Not applicable (no async operations in package).

**Error Testing:**
Pattern - use `expect_error()` and `expect_message()`:
```r
test_that("backend validation catches errors", {
  # Test missing files
  expect_error(
    nifti_backend(
      source = "/nonexistent/file.nii",
      mask_source = "/nonexistent/mask.nii"
    ),
    "Source files not found"
  )

  # Test invalid parameters
  expect_error(
    matrix_backend(
      data_matrix = "not a matrix",
      mask = rep(TRUE, 10)
    ),
    "data_matrix must be a matrix"
  )
})
```

**Conditional Tests (skip_if):**
Pattern - skip when dependencies unavailable:
```r
test_that("acquisition_onsets function works", {
  skip_if_not(exists("acquisition_onsets"), "acquisition_onsets not available")
  onsets <- acquisition_onsets(sf)
  expect_equal(length(onsets), 60)
})
```

**Data-Driven Tests:**
Use iteration to test multiple scenarios:
```r
test_that("chunking strategies produce complete coverage", {
  for (nchunks in c(1, 2, 5, 10, 100)) {
    chunks <- data_chunks(dataset, nchunks = nchunks)
    # Verify coverage for each strategy
    expect_equal(
      sort(unique(all_voxel_inds)),
      1:n_voxels
    )
  }
})
```

**Mocking Async/External Calls:**
File I/O mocked with temporary files:
```r
temp_files <- c(
  tempfile(fileext = ".nii"),
  tempfile(fileext = ".nii"),
  tempfile(fileext = ".nii")
)
temp_mask <- tempfile(fileext = ".nii")

# Create actual temp files
for (f in c(temp_files, temp_mask)) {
  file.create(f)
}

# ...test...

# Cleanup
unlink(c(temp_files, temp_mask))
```

## Test Quality

**Best Practices Applied:**
1. Clear test names describing what is tested
2. Arrange-Act-Assert (AAA) pattern within tests
3. One logical concept per `test_that()` block
4. Reusable test data setup
5. Golden tests for regression detection
6. Comprehensive error scenario coverage
7. Snapshot testing for output validation
8. Conditional skips for optional dependencies

**Coverage Stats:**
- 40+ test files
- ~11,000 lines of test code
- Coverage target: 70% (patch), auto (project)
- All major backends tested
- Error conditions comprehensively tested

---

*Testing analysis: 2026-01-22*
