# Phase 4: Test Coverage - Research

**Researched:** 2026-01-22
**Domain:** R package testing with testthat, backend testing patterns, optional dependency handling
**Confidence:** HIGH

## Summary

Achieving 80%+ test coverage for R packages requires systematic testing of core functionality, edge cases, and error paths using testthat 3.3.2 and covr. For backend implementations like zarr_backend and h5_backend, the key challenges are handling optional dependencies gracefully with `skip_if_not_installed()` and generating minimal synthetic test data on-the-fly using withr's temporary directory management.

The fmridataset package already has solid testing infrastructure with helper functions and test patterns established. The focus is on expanding coverage for specific files (zarr_backend.R at 5%, h5_backend.R at 26%, as_delayed_array.R at 10%) by testing untested code paths, error conditions, and edge cases. Tests should be hermetic (self-contained), fast (use tiny data dimensions like 4x4x4x10), and skip gracefully when optional packages (hdf5r, zarr, DelayedArray) are unavailable.

**Primary recommendation:** Use withr::local_tempdir() for automatic cleanup, generate test data programmatically with minimal dimensions, test all backend methods systematically (constructor, open, close, get_dims, get_mask, get_data, get_metadata), and verify error conditions alongside happy paths.

## Standard Stack

The established libraries/tools for R package testing:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| testthat | 3.3.2 | Unit testing framework | Official R-lib testing framework, industry standard |
| covr | 3.6.4+ | Test coverage measurement | De facto standard for coverage tracking, integrates with CI/CD |
| withr | 3.0.2+ | Temporary state management | Recommended for self-cleaning test fixtures |
| mockery | 0.4.4+ | Mocking framework | Integration with testthat for dependency mocking |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hdf5r | Latest | HDF5 file I/O | Testing h5_backend (optional dependency) |
| zarr | 0.1.1+ | Zarr format I/O | Testing zarr_backend (optional dependency) |
| neuroim2 | Package dep | Neuroimaging objects | Creating mock NeuroVec objects for tests |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| testthat | RUnit | testthat has better ecosystem integration, modern features |
| covr | testCoverage | covr has better reporting, CI/CD integration |
| withr | manual cleanup | withr guarantees cleanup even on error |

**Installation:**
```r
# Testing infrastructure (in Suggests)
install.packages(c("testthat", "covr", "withr", "mockery"))

# Optional backend dependencies
install.packages(c("hdf5r", "zarr"))
```

## Architecture Patterns

### Recommended Test File Structure
```
tests/testthat/
├── helper-*.R          # Helper functions (loaded before tests)
├── setup-*.R           # Setup code run once
├── test-*.R            # Test files
└── _snaps/             # Snapshot test data
```

### Pattern 1: Backend Testing with Optional Dependencies
**What:** Test backend implementations that depend on optional packages
**When to use:** Testing zarr_backend, h5_backend, or any backend requiring non-core dependencies
**Example:**
```r
# Source: testthat documentation, existing test_zarr_backend.R
test_that("zarr_backend opens and reads data", {
  skip_if_not_installed("zarr")

  # Generate test data
  tmp_dir <- withr::local_tempdir()
  arr <- array(rnorm(4 * 4 * 4 * 10), dim = c(4, 4, 4, 10))
  zarr::as_zarr(arr, location = file.path(tmp_dir, "test.zarr"))

  # Test backend
  backend <- zarr_backend(file.path(tmp_dir, "test.zarr"))
  backend <- backend_open(backend)

  # Verify dimensions
  dims <- backend_get_dims(backend)
  expect_equal(dims$spatial, c(4, 4, 4))
  expect_equal(dims$time, 10)

  # Test data access
  data <- backend_get_data(backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(data), c(5, 10))

  # Cleanup handled by withr::local_tempdir()
})
```

### Pattern 2: Helper Function for Test Data Generation
**What:** Shared functions to create minimal test data in helper files
**When to use:** When multiple test files need similar synthetic data
**Example:**
```r
# Source: testthat best practices, helper-golden.R pattern
# In tests/testthat/helper-backends.R

#' Create minimal HDF5 test file
#' @param dims Dimensions (default: c(4, 4, 4, 10) for fast tests)
create_test_h5 <- function(dims = c(4, 4, 4, 10), path = NULL) {
  skip_if_not_installed("hdf5r")
  skip_if_not_installed("fmristore")

  if (is.null(path)) {
    path <- tempfile(fileext = ".h5")
  }

  # Generate data
  data_array <- array(rnorm(prod(dims)), dim = dims)

  # Write H5 file using hdf5r
  h5file <- hdf5r::H5File$new(path, mode = "w")
  h5file$create_dataset("data", data_array, chunk_dims = c(4, 4, 4, 1))
  h5file$close_all()

  path
}

#' Create minimal Zarr test store
create_test_zarr <- function(dims = c(4, 4, 4, 10), path = NULL) {
  skip_if_not_installed("zarr")

  if (is.null(path)) {
    path <- tempfile(pattern = "zarr_")
  }

  data_array <- array(rnorm(prod(dims)), dim = dims)
  zarr::as_zarr(data_array, location = path)

  path
}
```

### Pattern 3: Testing Error Conditions
**What:** Verify that functions fail appropriately with clear error messages
**When to use:** Testing input validation, missing files, invalid state
**Example:**
```r
# Source: R testing best practices, Advanced R Conditions chapter
test_that("backend validates inputs correctly", {
  # Invalid source type
  expect_error(
    zarr_backend(NULL),
    "source must be a single character string"
  )

  expect_error(
    zarr_backend(c("path1", "path2")),
    "source must be a single character string"
  )
})

test_that("backend requires package", {
  # Mock missing package
  with_mocked_bindings(
    requireNamespace = function(...) FALSE,
    .package = "base",
    {
      expect_error(
        zarr_backend("dummy.zarr"),
        "zarr package is required"
      )
    }
  )
})

test_that("backend handles missing files", {
  skip_if_not_installed("zarr")

  backend <- zarr_backend("/nonexistent/path.zarr")
  expect_error(
    backend_open(backend),
    "Zarr store not found"
  )
})
```

### Pattern 4: S4 Method Testing (for as_delayed_array)
**What:** Test S4 method dispatch and DelayedArray integration
**When to use:** Testing as_delayed_array.R and as_delayed_array_dataset.R
**Example:**
```r
# Source: testthat S4 testing documentation
test_that("as_delayed_array converts backend to DelayedArray", {
  skip_if_not_installed("DelayedArray")

  # Create test backend
  backend <- matrix_backend(matrix(1:20, nrow = 5, ncol = 4))

  # Convert to DelayedArray
  da <- as_delayed_array(backend)

  # Verify S4 class
  expect_s4_class(da, "DelayedArray")

  # Verify dimensions
  expect_equal(dim(da), c(5, 4))

  # Verify data matches
  expect_equal(as.matrix(da), backend$data_matrix)

  # Test subsetting
  sub <- da[2:4, 2:3]
  expect_equal(as.matrix(sub), backend$data_matrix[2:4, 2:3])
})
```

### Pattern 5: Coverage-Driven Test Expansion
**What:** Use covr to identify untested lines, write focused tests
**When to use:** Hitting specific coverage targets like 80%
**Example:**
```r
# Workflow for increasing coverage:

# 1. Generate coverage report for single file
cov <- covr::file_coverage("R/zarr_backend.R", "tests/testthat/test_zarr_backend.R")
covr::report(cov)  # Visual HTML report

# 2. Identify untested lines with zero_coverage()
uncovered <- covr::zero_coverage(cov)
print(uncovered)

# 3. Write tests targeting specific uncovered code paths
test_that("zarr_backend handles preload option", {
  # Tests line 134-137 (preload branch)
  skip_if_not_installed("zarr")

  tmp_dir <- withr::local_tempdir()
  arr <- array(rnorm(2 * 3 * 2 * 2), dim = c(2, 3, 2, 2))
  zarr::as_zarr(arr, location = file.path(tmp_dir, "test.zarr"))

  backend <- zarr_backend(file.path(tmp_dir, "test.zarr"), preload = TRUE)
  expect_message(
    backend <- backend_open(backend),
    "Preloading Zarr data"
  )

  expect_false(is.null(backend$data_cache))
})

# 4. Re-run coverage to verify improvement
cov_after <- covr::file_coverage("R/zarr_backend.R", "tests/testthat/test_zarr_backend.R")
covr::percent_coverage(cov_after)  # Should be higher
```

### Anti-Patterns to Avoid
- **Large test fixtures:** Don't commit large test data files; generate programmatically
- **Environment-specific paths:** Don't hardcode paths; use tempfile() and withr::local_tempdir()
- **Skipping too broadly:** Don't skip entire test files; skip individual tests that need optional deps
- **Testing implementation details:** Test the public API, not internal helper functions
- **Over-mocking:** Mock only external dependencies, not your own code under test

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Temporary file cleanup | Manual on.exit() per test | `withr::local_tempdir()` | Guaranteed cleanup even on error, simpler syntax |
| Coverage measurement | Parsing R code for executed lines | `covr::package_coverage()` | Handles compiled code, accurate line tracking, CI integration |
| Skipping tests conditionally | `if (requireNamespace(...))` | `skip_if_not_installed()` | Proper skip reporting, standardized messages |
| Mocking package availability | Reassigning functions manually | `with_mocked_bindings()` | Clean namespace handling, automatic restoration |
| Test data fixtures | Committed .rds files | Generate in helper functions | Keeps repo small, flexible dimensions, version control friendly |

**Key insight:** testthat and withr provide battle-tested infrastructure for test isolation and cleanup. The 3.x versions introduced hermetic test principles that make tests more reliable and easier to debug.

## Common Pitfalls

### Pitfall 1: Tests Fail When Optional Packages Missing
**What goes wrong:** Tests error out instead of skipping gracefully when hdf5r/zarr unavailable
**Why it happens:** Forgetting `skip_if_not_installed()` at test level, not just file level
**How to avoid:** Add skip guard to every test that uses optional dependency
**Warning signs:** CI failures on minimal dependency builds, "package not available" errors

```r
# BAD: Will fail if zarr not installed
test_that("zarr backend works", {
  backend <- zarr_backend("test.zarr")  # Error if zarr missing
  ...
})

# GOOD: Skips gracefully
test_that("zarr backend works", {
  skip_if_not_installed("zarr")  # Add this first
  backend <- zarr_backend("test.zarr")
  ...
})
```

### Pitfall 2: Temp Files Not Cleaned Up
**What goes wrong:** Test temp files accumulate in /tmp, eventually fill disk
**Why it happens:** Using tempfile() without cleanup, or cleanup code not reached on error
**How to avoid:** Use `withr::local_tempdir()` which guarantees cleanup
**Warning signs:** Growing /tmp directory, "no space left" errors

```r
# BAD: No cleanup if test fails
test_that("backend reads file", {
  tmp_file <- tempfile()
  # Create file, test it
  unlink(tmp_file)  # Never reached if test errors
})

# GOOD: Automatic cleanup
test_that("backend reads file", {
  tmp_dir <- withr::local_tempdir()
  tmp_file <- file.path(tmp_dir, "test.h5")
  # Create file, test it
  # Cleanup happens automatically
})
```

### Pitfall 3: Large Test Data Slows Tests
**What goes wrong:** Tests take minutes to run, discouraging frequent testing
**Why it happens:** Using realistic data dimensions (64x64x64x200) instead of minimal test dimensions
**How to avoid:** Use tiny dimensions (4x4x4x10) that exercise logic without computation overhead
**Warning signs:** `R CMD check` times out, developers skip running tests locally

```r
# BAD: Large realistic dimensions
arr <- array(rnorm(64 * 64 * 64 * 200), dim = c(64, 64, 64, 200))  # 10+ seconds

# GOOD: Minimal dimensions
arr <- array(rnorm(4 * 4 * 4 * 10), dim = c(4, 4, 4, 10))  # <0.1 seconds
```

### Pitfall 4: Testing Backend Closes When Not Open
**What goes wrong:** backend_close() tested but backend_open() never verified it actually opened
**Why it happens:** Focusing on code coverage percentage instead of meaningful test paths
**How to avoid:** Test state transitions explicitly (closed → open → data access → close)
**Warning signs:** 80% coverage but bugs in open/close state management

```r
# BAD: Fake coverage without testing state
test_that("backend closes", {
  backend <- zarr_backend("test.zarr")
  expect_silent(backend_close(backend))  # Never opened!
})

# GOOD: Test actual state progression
test_that("backend lifecycle", {
  skip_if_not_installed("zarr")

  tmp_dir <- withr::local_tempdir()
  create_test_zarr_store(tmp_dir)

  backend <- zarr_backend(tmp_dir)
  expect_false(backend$is_open)

  backend <- backend_open(backend)
  expect_true(backend$is_open)

  data <- backend_get_data(backend)
  expect_type(data, "double")

  backend_close(backend)
  expect_false(backend$is_open)
})
```

### Pitfall 5: Assuming DelayedArray is Always Available
**What goes wrong:** Tests fail with "package 'DelayedArray' not found" despite skip guards
**Why it happens:** DelayedArray loaded at top level before skip_if_not_installed() runs
**How to avoid:** Keep skip guards before any code that requires the package
**Warning signs:** Tests pass locally (you have DelayedArray) but fail on CI

```r
# BAD: DelayedArray loaded before skip
test_that("as_delayed_array works", {
  da <- DelayedArray::DelayedArray(...)  # Fails if not installed
  skip_if_not_installed("DelayedArray")  # Too late!
})

# GOOD: Skip first
test_that("as_delayed_array works", {
  skip_if_not_installed("DelayedArray")
  da <- DelayedArray::DelayedArray(...)  # Safe
})
```

### Pitfall 6: Coverage Inflation from Untested Error Paths
**What goes wrong:** 80% coverage achieved but critical error handling never tested
**Why it happens:** Stop() and error messages count as "covered" when functions run successfully
**How to avoid:** Explicitly test error conditions with expect_error()
**Warning signs:** High coverage but production errors reveal untested edge cases

```r
# Tests "cover" error path but never verify it works
test_that("backend opens file", {
  skip_if_not_installed("zarr")
  tmp_dir <- withr::local_tempdir()
  create_test_zarr_store(tmp_dir)

  backend <- zarr_backend(tmp_dir)
  backend <- backend_open(backend)  # Success path
  expect_true(backend$is_open)
})

# Should ALSO test error path explicitly
test_that("backend errors on missing file", {
  skip_if_not_installed("zarr")

  backend <- zarr_backend("/nonexistent/path.zarr")
  expect_error(
    backend_open(backend),
    "Zarr store not found"
  )
})
```

## Code Examples

Verified patterns from official sources:

### Testing Backend Constructor
```r
# Source: test_zarr_backend.R, testthat documentation
test_that("backend constructor validates inputs", {
  skip_if_not_installed("zarr")

  # Test invalid types
  expect_error(zarr_backend(NULL), "source must be")
  expect_error(zarr_backend(c("a", "b")), "source must be")
  expect_error(zarr_backend(123), "source must be")

  # Test valid construction
  backend <- zarr_backend("test.zarr")
  expect_s3_class(backend, "zarr_backend")
  expect_s3_class(backend, "storage_backend")
  expect_equal(backend$source, "test.zarr")
  expect_false(backend$is_open)
})
```

### Testing Backend Methods Systematically
```r
# Source: Backend testing best practices
test_that("backend implements all required methods", {
  skip_if_not_installed("hdf5r")
  skip_if_not_installed("fmristore")

  # Setup
  tmp_dir <- withr::local_tempdir()
  h5_path <- create_test_h5(dims = c(4, 4, 4, 10), path = file.path(tmp_dir, "test.h5"))
  mask_path <- create_test_mask_h5(dims = c(4, 4, 4), path = file.path(tmp_dir, "mask.h5"))

  backend <- h5_backend(h5_path, mask_path)

  # Test backend_open
  backend <- backend_open(backend)
  expect_true(!is.null(backend$h5_objects))

  # Test backend_get_dims
  dims <- backend_get_dims(backend)
  expect_equal(dims$spatial, c(4, 4, 4))
  expect_equal(dims$time, 10)

  # Test backend_get_mask
  mask <- backend_get_mask(backend)
  expect_type(mask, "logical")
  expect_equal(length(mask), 4 * 4 * 4)
  expect_true(sum(mask) > 0)

  # Test backend_get_data (full)
  data_full <- backend_get_data(backend)
  expect_equal(dim(data_full), c(10, sum(mask)))

  # Test backend_get_data (subset)
  data_sub <- backend_get_data(backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(data_sub), c(5, 10))

  # Test backend_get_metadata
  meta <- backend_get_metadata(backend)
  expect_type(meta, "list")
  expect_true("format" %in% names(meta))

  # Test backend_close
  backend_close(backend)
  # Could verify h5 handles closed if needed
})
```

### Testing with Mocked Dependencies
```r
# Source: testthat mocking documentation
test_that("backend handles missing package gracefully", {
  with_mocked_bindings(
    requireNamespace = function(pkg, ...) {
      if (pkg == "zarr") FALSE else TRUE
    },
    .package = "base",
    {
      expect_error(
        zarr_backend("test.zarr"),
        "zarr package is required"
      )
    }
  )
})

test_that("backend handles broken package installation", {
  skip_if_not_installed("hdf5r")

  # Simulate hdf5r installed but broken (load fails)
  # Let test fail - this is a real problem
  with_mocked_bindings(
    H5NeuroVec = function(...) stop("hdf5r library linking error"),
    .package = "fmristore",
    {
      tmp_dir <- withr::local_tempdir()
      h5_path <- file.path(tmp_dir, "test.h5")

      expect_error(
        backend <- h5_backend(h5_path, h5_path),
        "Failed to load H5NeuroVec"
      )
    }
  )
})
```

### Testing dataset_methods.R Delegation
```r
# Source: dataset_methods.R structure
test_that("dataset methods delegate to sampling_frame", {
  # Create dataset with known sampling frame
  mat <- matrix(rnorm(100), nrow = 10, ncol = 10)
  dataset <- matrix_dataset(mat, TR = 2.0, run_length = 10)

  # Test each delegated method
  expect_equal(get_TR(dataset), 2.0)
  expect_equal(n_timepoints(dataset), 10)
  expect_equal(n_runs(dataset), 1)
  expect_equal(get_run_lengths(dataset), 10)
  expect_equal(get_total_duration(dataset), 20.0)  # 10 * 2.0

  # Test multi-run case
  dataset_multi <- matrix_dataset(mat, TR = 2.0, run_length = c(5, 5))
  expect_equal(n_runs(dataset_multi), 2)
  expect_equal(get_run_lengths(dataset_multi), c(5, 5))
  expect_equal(blocklens(dataset_multi), c(5, 5))
})
```

### Helper Function Template
```r
# Source: helper-golden.R pattern, withr documentation
# In tests/testthat/helper-backends.R

#' Create test Zarr store with minimal dimensions
#'
#' @param dims Array dimensions (default: c(4, 4, 4, 10))
#' @param path Directory path (default: temp directory)
#' @return Path to Zarr store
create_test_zarr <- function(dims = c(4, 4, 4, 10), path = NULL) {
  skip_if_not_installed("zarr")

  if (is.null(path)) {
    # Caller should manage temp directory with withr::local_tempdir()
    path <- tempfile(pattern = "zarr_test_")
  }

  # Generate random data
  data_array <- array(rnorm(prod(dims)), dim = dims)

  # Write Zarr store
  zarr::as_zarr(data_array, location = path)

  invisible(path)
}

#' Create test HDF5 file with minimal dimensions
#'
#' @param dims Array dimensions (default: c(4, 4, 4, 10))
#' @param path File path (default: temp file)
#' @return Path to H5 file
create_test_h5 <- function(dims = c(4, 4, 4, 10), path = NULL) {
  skip_if_not_installed("hdf5r")

  if (is.null(path)) {
    path <- tempfile(fileext = ".h5")
  }

  # Generate data
  data_array <- array(rnorm(prod(dims)), dim = dims)

  # Write H5 file
  h5file <- hdf5r::H5File$new(path, mode = "w")
  h5file$create_dataset("data", data_array, chunk_dims = c(4, 4, 4, 1))
  h5file$close_all()

  invisible(path)
}

#' Create test mask (H5 or array)
create_test_mask_h5 <- function(dims = c(4, 4, 4), path = NULL, all_valid = TRUE) {
  skip_if_not_installed("hdf5r")

  if (is.null(path)) {
    path <- tempfile(fileext = ".h5")
  }

  # Generate mask
  if (all_valid) {
    mask_array <- array(TRUE, dim = dims)
  } else {
    mask_array <- array(sample(c(TRUE, FALSE), prod(dims), replace = TRUE), dim = dims)
  }

  # Write mask to H5
  h5file <- hdf5r::H5File$new(path, mode = "w")
  h5file$create_dataset("data/elements", mask_array + 0L)  # Convert to integer
  h5file$close_all()

  invisible(path)
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| testthat 2.x with context() | testthat 3.x with test_that() only | testthat 3.0.0 (2020) | Simpler structure, better hermetic tests |
| Manual cleanup with on.exit() | withr::local_tempdir() | withr 2.0+ | Guaranteed cleanup, cleaner code |
| with_mock() for mocking | with_mocked_bindings() | testthat 3.0.0 | More reliable, works with base functions |
| Committed test fixtures | Generated test data | Best practice evolution | Smaller repos, flexible test dimensions |
| Skip entire test files | Skip individual tests | testthat 3.0.0 skip reporting | Better visibility into what's skipped |

**Deprecated/outdated:**
- `context()`: Removed in testthat 3.0.0, replaced by better test organization
- `with_mock()`: Defunct in testthat 3.2.0, replaced by `with_mocked_bindings()`
- `local_mock()`: Defunct, replaced by `local_mocked_bindings()`

## Open Questions

Things that couldn't be fully resolved:

1. **DelayedArray Testing Strategy Without Bioconductor**
   - What we know: DelayedArray is fundamentally a Bioconductor package with deep dependencies (BiocGenerics, S4Vectors, IRanges)
   - What's unclear: Whether as_delayed_array.R testing should treat DelayedArray as always-optional or assume it's available in most contexts
   - Recommendation: Use `skip_if_not_installed("DelayedArray")` in all tests. Package already has `.ensure_delayed_array()` guard. Tests verify the wrapping logic when DelayedArray is present; skip gracefully when absent.

2. **H5 Backend Testing with Broken hdf5r**
   - What we know: Context says "if hdf5r is installed but broken (load errors), let tests fail"
   - What's unclear: How to distinguish "not installed" (skip) from "installed but broken" (fail)
   - Recommendation: Use `skip_if_not_installed("hdf5r")` normally. If hdf5r loads but fmristore::H5NeuroVec() fails, let the test error propagate - this surfaces a real system issue.

3. **Zarr v2 vs v3 Testing**
   - What we know: CRAN zarr package only supports Zarr v3 format
   - What's unclear: Should tests explicitly verify rejection of v2 stores?
   - Recommendation: Test local Zarr v3 stores only (as decided). If test data accidentally uses v2 format, let natural error occur. No need for explicit v2 rejection tests since limitation is documented.

4. **Optimal Test Data Dimensions**
   - What we know: Context recommends ~4x4x4 volume, ~10 timepoints
   - What's unclear: Whether different backends need different dimensions (e.g., test chunking logic)
   - Recommendation: Default to c(4, 4, 4, 10) in helper functions, allow override via parameters. Some specific tests (chunking strategy, large subset behavior) can use slightly larger dimensions (8x8x8x20) if needed to trigger code paths.

## Sources

### Primary (HIGH confidence)
- [testthat 3.3.2 documentation](https://cran.r-project.org/web/packages/testthat/testthat.pdf) - Latest version (January 11, 2026)
- [Skipping tests • testthat](https://testthat.r-lib.org/articles/skipping.html) - Official guidance on skip_if_not_installed()
- [Test Coverage for Packages • covr](https://covr.r-lib.org/) - Coverage measurement and reporting
- [withr package documentation](https://testthat.r-lib.org/articles/special-files.html) - Temporary file management
- [Mocking • testthat](https://testthat.r-lib.org/articles/mocking.html) - with_mocked_bindings() usage
- [R Packages (2e) - Testing Design](https://r-pkgs.org/testing-design.html) - Hadley Wickham's testing best practices

### Secondary (MEDIUM confidence)
- [Helper code and files for testthat tests - R-hub blog](https://blog.r-hub.io/2020/11/18/testthat-utility-belt/) - Helper function patterns
- [Advanced R - Conditions](https://adv-r.hadley.nz/conditions.html) - Error handling in R
- [hdf5r documentation](https://hhoeflin.github.io/hdf5r/) - HDF5 file interface for R

### Tertiary (LOW confidence)
- [What unit test coverage percentage should teams aim for? | TechTarget](https://www.techtarget.com/searchsoftwarequality/tip/What-unit-test-coverage-percentage-should-teams-aim-for) - 80% coverage standard discussion
- [A Comparison of HDF5, Zarr, and netCDF4](https://arxiv.org/pdf/2207.09503) - Test data generation patterns for array formats

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - testthat 3.3.2 and covr are authoritative, well-documented
- Architecture patterns: HIGH - Based on official testthat documentation and existing codebase patterns
- Helper functions: HIGH - Verified against existing helper-golden.R and testthat best practices
- Backend testing: HIGH - Based on existing test_zarr_backend.R and official documentation
- Edge cases: MEDIUM - Best practices synthesized from multiple sources
- Pitfalls: HIGH - Based on official testthat migration guides and common patterns

**Research date:** 2026-01-22
**Valid until:** ~30 days (testthat/covr are stable; check for patches)

**Key findings:**
1. testthat 3.3.2 (Jan 2026) is current - use with_mocked_bindings() not deprecated with_mock()
2. withr::local_tempdir() is recommended over manual cleanup
3. 80% coverage is industry standard target - balance thoroughness with pragmatism
4. skip_if_not_installed() should be in every test that uses optional deps
5. Generate test data with minimal dimensions (4x4x4x10) for speed
6. Helper functions in helper-backends.R for shared test data generation
7. Test all backend methods systematically: constructor, open, close, get_dims, get_mask, get_data, get_metadata
8. Explicitly test error conditions, not just success paths
9. DelayedArray testing requires skip guards - it's a Bioconductor package
10. Use covr::file_coverage() and zero_coverage() to identify gaps
