# Phase 2: Tech Debt - Research

**Researched:** 2026-01-22
**Domain:** R resource management, HDF5 file handle cleanup, S3 method validation
**Confidence:** HIGH

## Summary

This phase addresses critical resource leaks in the H5 backend and commits a pending S3 method validation fix. The research investigated R's resource management patterns, specifically `on.exit()` for cleanup, HDF5 file handle management with hdf5r, and S3 method introspection with `utils::getS3method()`.

**Current Issues Identified:**
1. **backend_get_dims.h5_backend**: Creates H5NeuroVec objects in lines 216, 223 that won't be closed if `dim()` or other operations fail before explicit `close()` calls
2. **backend_get_mask.h5_backend**: Creates H5NeuroVol (line 274) and H5NeuroVec (line 280) that won't be closed if `as.array()`, `space()`, or subsequent operations fail
3. **backend_get_data.h5_backend**: Creates H5NeuroVec objects (line 346) without `on.exit()` protection, relies on cleanup at line 382 which won't execute if `series()` fails
4. **storage_backend.R**: Uses string concatenation to check method existence instead of `utils::getS3method()` (already fixed in working tree, needs commit)

**Primary recommendation:** Add `on.exit(close(handle), add = TRUE, after = FALSE)` immediately after every H5 object creation to ensure cleanup occurs even when errors happen in subsequent operations. This is the standard R pattern for resource management and aligns with hdf5r's explicit closure requirements.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| base::on.exit | R 4.3+ | Resource cleanup | Built-in R mechanism for guaranteed cleanup regardless of exit path (normal/error) |
| utils::getS3method | R 4.3+ | S3 method introspection | Official R function for checking S3 method existence, handles namespace visibility |
| hdf5r | Suggested dep | HDF5 file access | Already used by fmristore, provides explicit close() methods |
| testthat | 3.3.2 | Testing framework | Package standard, current version (Jan 2026) |
| withr | 3.0.2+ | Test fixtures | testthat dependency, provides `defer()` for test cleanup |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fmristore | Current | H5NeuroVec/H5NeuroVol wrappers | Already package dependency for H5 backend |
| mockery/mockr | Suggested | Mocking for tests | Already available, can mock error conditions |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| on.exit() | tryCatch(..., finally = {}) | `finally` only scopes to that block; `on.exit()` is cleaner for function-level resource management |
| utils::getS3method | exists(paste0(generic, ".", class)) | String approach fails with namespaced methods, doesn't respect S3 dispatch rules |
| Explicit close() | R garbage collection | Timing unpredictable, can prevent file reopening, not reliable |

**Installation:**
None required - all core libraries are base R or existing package dependencies.

## Architecture Patterns

### Recommended Resource Management Structure
```r
# Pattern 1: Function opens, uses, closes resource
backend_get_something.h5_backend <- function(backend) {
  if (cached) return(cached)

  if (is.character(backend$source)) {
    tryCatch(
      {
        h5_handle <- fmristore::H5NeuroVec(path, ...)
        on.exit(close(h5_handle), add = TRUE, after = FALSE)

        # Use h5_handle safely - cleanup guaranteed
        data <- extract_data(h5_handle)
        result <- process_data(data)
        result
      },
      error = function(e) {
        stop_fmridataset(...)
      }
    )
  }
}
```

### Pattern 1: Single Resource Acquisition
**What:** Open resource, register cleanup, use resource
**When to use:** Single H5 file needs to be opened and closed within function scope

**Example:**
```r
# Source: Current codebase pattern with on.exit() added
backend_get_metadata.h5_backend <- function(backend) {
  h5_obj <- if (!is.null(backend$h5_objects)) {
    backend$h5_objects[[1]]
  } else if (is.character(backend$source)) {
    first_h5 <- fmristore::H5NeuroVec(backend$source[1], dataset_name = backend$data_dataset)
    on.exit(close(first_h5), add = TRUE, after = FALSE)  # ADD THIS
    first_h5
  } else {
    backend$source[[1]]
  }

  # Extract metadata - cleanup guaranteed by on.exit
  space_obj <- space(h5_obj)
  list(format = "h5", affine = trans(space_obj), ...)
}
```

### Pattern 2: Multiple Resources in Loop
**What:** Open multiple resources sequentially, ensure each closes even if later ones fail
**When to use:** Iterating over multiple H5 files (e.g., calculating total time dimension)

**Example:**
```r
# Source: R manual on.exit documentation (stat.ethz.ch)
# Adapted for H5 file iteration
total_time <- sum(sapply(backend$source, function(file_path) {
  h5_obj <- fmristore::H5NeuroVec(file_path, dataset_name = backend$data_dataset)
  on.exit(close(h5_obj), add = TRUE, after = FALSE)  # ADD THIS
  time_dim <- dim(h5_obj)[4]
  time_dim
}))
```

### Pattern 3: Error Re-wrapping with Cleanup
**What:** Catch H5/system errors, ensure cleanup, re-throw with domain-specific error
**When to use:** All H5 operations that can fail (file not found, corrupt data, etc.)

**Example:**
```r
# Source: Current codebase error handling pattern
tryCatch(
  {
    first_h5 <- fmristore::H5NeuroVec(backend$source[1], ...)
    on.exit(close(first_h5), add = TRUE, after = FALSE)  # ADD THIS
    d <- dim(first_h5)
    # ... use d ...
  },
  error = function(e) {
    # on.exit cleanup happens automatically here before error propagates
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf("Failed to read H5 dimensions: %s", e$message),
      file = backend$source[1],
      operation = "read_header"
    )
  }
)
```

### Pattern 4: S3 Method Validation
**What:** Check if S3 method exists using proper introspection
**When to use:** Backend validation, ensuring contract implementation

**Example:**
```r
# Source: Advanced R by Hadley Wickham (https://adv-r.hadley.nz/s3.html)
# utils::getS3method documentation (stat.ethz.ch)
for (method in required_methods) {
  method_impl <- utils::getS3method(method, backend_class, optional = TRUE)
  if (is.null(method_impl)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Backend class '%s' must implement method '%s'", backend_class, method)
    )
  }
}
```

### Anti-Patterns to Avoid
- **Close without on.exit:** Relying on explicit `close()` calls after operations that can fail leaves dangling handles
- **on.exit without add = TRUE:** Overwrites previous handlers; multiple resources won't all clean up
- **Trusting garbage collection:** HDF5 files can remain locked, preventing reopening (hdf5r vignette warning)
- **String-based S3 method checking:** `exists(paste0(method, ".", class))` fails for namespaced methods

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Resource cleanup on error | Manual try/catch with cleanup code | `on.exit(cleanup, add = TRUE, after = FALSE)` | Handles all exit paths automatically (error, return, stop), multiple handlers supported |
| S3 method existence checking | String concatenation + exists() | `utils::getS3method(method, class, optional = TRUE)` | Respects namespaces, follows S3 dispatch rules, official R mechanism |
| Test resource cleanup | Manual cleanup in each test | `withr::defer(cleanup, teardown_env())` | Guaranteed cleanup even if test fails, more flexible than on.exit for tests |
| HDF5 file closure | Rely on R garbage collection | Explicit `close()` with `on.exit()` | Garbage collection timing unpredictable, can prevent file reopening |

**Key insight:** Resource management in R looks deceptively simple (just call `close()`), but error paths create numerous leak scenarios. The `on.exit()` pattern is battle-tested across all major R packages and handles edge cases (multiple resources, nested calls, errors in cleanup itself).

## Common Pitfalls

### Pitfall 1: Close After Operations That Can Fail
**What goes wrong:** Code pattern `handle <- open(); data <- extract(handle); close(handle)` leaks handle if `extract()` fails
**Why it happens:** R doesn't have automatic destructors like C++ RAII; explicit cleanup happens only if execution reaches that line
**How to avoid:** Place `on.exit(close(handle), add = TRUE, after = FALSE)` immediately after opening
**Warning signs:**
- `close()` calls not immediately after handle creation
- Operations between handle creation and `close()` that can fail (dim, as.array, space, series)
- tryCatch blocks where cleanup is outside the error handler

**Example from codebase:**
```r
# BEFORE (lines 216-218 in h5_backend.R)
first_h5 <- fmristore::H5NeuroVec(backend$source[1], dataset_name = backend$data_dataset)
d <- dim(first_h5)  # If this fails, first_h5 never closed!
close(first_h5)

# AFTER
first_h5 <- fmristore::H5NeuroVec(backend$source[1], dataset_name = backend$data_dataset)
on.exit(close(first_h5), add = TRUE, after = FALSE)
d <- dim(first_h5)  # Cleanup guaranteed even if dim() fails
```

### Pitfall 2: Multiple Resources Without add = TRUE
**What goes wrong:** Second `on.exit()` call overwrites first, so first resource never cleaned up
**Why it happens:** Default `add = FALSE` replaces previous handler
**How to avoid:** Always use `add = TRUE` even if only one resource currently (future-proofs code)
**Warning signs:**
- Multiple resources in same function
- on.exit() called without explicit `add = TRUE`

### Pitfall 3: Nested Resource Operations
**What goes wrong:** Inner loop opens H5 files, outer code creates H5 object, errors in inner loop prevent outer cleanup
**Why it happens:** Cleanup order matters; `after = FALSE` ensures LIFO (last opened, first closed)
**How to avoid:** Use `after = FALSE` to ensure reverse-order cleanup (like C++ stack unwinding)
**Warning signs:**
- sapply/lapply over files that open H5 handles
- Multiple H5 objects created in same scope

**Example from codebase:**
```r
# backend_get_dims lines 221-228 has nested opening
first_h5 <- fmristore::H5NeuroVec(backend$source[1], ...)  # Outer
on.exit(close(first_h5), add = TRUE, after = FALSE)

total_time <- sum(sapply(backend$source, function(file_path) {
  h5_obj <- fmristore::H5NeuroVec(file_path, ...)  # Inner loop
  on.exit(close(h5_obj), add = TRUE, after = FALSE)
  dim(h5_obj)[4]
}))
```

### Pitfall 4: Silent Errors in Cleanup
**What goes wrong:** If `close()` itself fails, can mask original error or leave resource open
**Why it happens:** Errors in cleanup handlers propagate
**How to avoid:** For non-critical cleanup, use `tryCatch(close(obj), error = function(e) invisible(NULL))`; for critical cleanup, let it fail loudly
**Warning signs:**
- Already partially closed handles
- Corrupt H5 files
- Permission errors

**Current pattern in codebase (backend_close.h5_backend line 196-198):**
```r
lapply(backend$h5_objects, function(obj) {
  tryCatch(close(obj), error = function(e) invisible(NULL))
})
```
This is correct for batch cleanup where you want to try closing all handles.

## Code Examples

Verified patterns from official sources:

### Canonical on.exit Pattern
```r
# Source: R manual on.exit documentation
# https://stat.ethz.ch/R-manual/R-devel/library/base/html/on.exit.html
backend_function <- function(backend) {
  resource <- acquire_resource()
  on.exit(release_resource(resource), add = TRUE, after = FALSE)

  # Use resource safely
  result <- process(resource)
  result
}
```

### H5 Handle Cleanup (Single File)
```r
# Source: Adapted from codebase + hdf5r vignette patterns
# https://cran.r-project.org/web/packages/hdf5r/vignettes/hdf5r.html
backend_get_dims.h5_backend <- function(backend) {
  if (!is.null(backend$dims)) return(backend$dims)

  if (is.character(backend$source)) {
    tryCatch({
      first_h5 <- fmristore::H5NeuroVec(backend$source[1], dataset_name = backend$data_dataset)
      on.exit(close(first_h5), add = TRUE, after = FALSE)

      d <- dim(first_h5)
      # ... rest of function
    }, error = function(e) {
      stop_fmridataset(fmridataset_error_backend_io, ...)
    })
  }
}
```

### H5 Handle Cleanup (Multiple Files in Loop)
```r
# Source: Adapted from codebase backend_get_dims.h5_backend
total_time <- if (length(backend$source) > 1) {
  sum(sapply(backend$source, function(file_path) {
    h5_obj <- fmristore::H5NeuroVec(file_path, dataset_name = backend$data_dataset)
    on.exit(close(h5_obj), add = TRUE, after = FALSE)
    time_dim <- dim(h5_obj)[4]
    time_dim
  }))
} else {
  d[4]
}
```

### S3 Method Validation
```r
# Source: Current fix in storage_backend.R (uncommitted)
# utils::getS3method documentation (stat.ethz.ch)
for (method in required_methods) {
  method_impl <- utils::getS3method(method, backend_class, optional = TRUE)
  if (is.null(method_impl)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Backend class '%s' must implement method '%s'", backend_class, method)
    )
  }
}
```

### Testing Resource Cleanup
```r
# Source: testthat test fixtures documentation
# https://testthat.r-lib.org/articles/test-fixtures.html
test_that("h5_backend cleans up handles on error", {
  skip_if_not_installed("withr")
  skip_if_not_installed("fmristore")

  # Create test file that will cause error
  temp_file <- withr::local_tempfile(fileext = ".h5")

  # Induce error in operation after opening
  expect_error(backend_get_dims(backend))

  # Verify cleanup occurred by checking no open handles remain
  # (Implementation depends on fmristore handle tracking)
})
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| String-based S3 method check | `utils::getS3method()` | R 3.0+ (formalized) | Handles namespaced methods correctly, respects S3 dispatch |
| `on.exit(expr)` | `on.exit(expr, add = TRUE, after = FALSE)` | R 2.15.0 (add param) | Enables multiple cleanup handlers, control over order |
| Implicit H5 finalization | Explicit close() calls | hdf5r design | Deterministic cleanup, prevents file locking issues |
| `on.exit()` in tests | `withr::defer()` | withr 2.0+ (2018) | More flexible environment control for test fixtures |

**Deprecated/outdated:**
- **exists(paste0(method, ".", class))**: Doesn't work with namespaced S3 methods, use `utils::getS3method()` instead
- **Relying on R garbage collection for H5 files**: hdf5r vignette explicitly warns against this, timing unpredictable

## Open Questions

Things that couldn't be fully resolved:

1. **Does fmristore provide handle tracking utilities?**
   - What we know: fmristore wraps hdf5r's H5NeuroVec/H5NeuroVol classes
   - What's unclear: Whether fmristore exposes any utilities to detect open handles (for testing cleanup)
   - Recommendation: Tests should focus on behavior (no errors on reopen) rather than internal handle state

2. **Should backend_open preload mode close handles after caching data?**
   - What we know: backend_open can preload H5NeuroVec objects into backend$h5_objects (lines 114-184)
   - What's unclear: Are these meant to stay open for repeated access, or should data be extracted and handles closed?
   - Recommendation: Current design keeps handles open for preload mode; this is fine if backend_close properly cleans up (it does, line 195-199)

3. **Test mock strategy for close() verification**
   - What we know: Current tests use mocks (test_h5_backend.R lines 12-18) with stub close methods
   - What's unclear: How to verify close was actually called with mocks
   - Recommendation: Use mockery package (already available) to track close() calls in error path tests

## Sources

### Primary (HIGH confidence)
- [R on.exit() documentation](https://stat.ethz.ch/R-manual/R-devel/library/base/html/on.exit.html) - Official R manual, definitive reference for add/after parameters
- [utils::getS3method documentation](https://stat.ethz.ch/R-manual/R-patched/library/utils/html/getS3method.html) - Official R manual for S3 method introspection
- [hdf5r package vignette](https://cran.r-project.org/web/packages/hdf5r/vignettes/hdf5r.html) - Official package documentation on close() vs close_all() and cleanup requirements
- Codebase files: R/h5_backend.R, R/storage_backend.R, tests/testthat/test_h5_backend.R - Current implementation to be fixed

### Secondary (MEDIUM confidence)
- [Advanced R - Conditions chapter](https://adv-r.hadley.nz/conditions.html) - Hadley Wickham's explanation of tryCatch vs on.exit, widely cited
- [testthat test fixtures](https://testthat.r-lib.org/articles/test-fixtures.html) - Official testthat documentation on withr::defer() for test cleanup
- [R-bloggers: Resource Connection Pattern](https://www.r-bloggers.com/2025/12/open-once-close-automatically-a-ressource-connection-pattern-for-r/) - Recent (Dec 2025) discussion of on.exit patterns in production systems

### Tertiary (LOW confidence)
- WebSearch results on "R on.exit error handling best practices" - General consensus across multiple sources supports add=TRUE, after=FALSE pattern

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools are base R or existing dependencies, well-documented
- Architecture: HIGH - Patterns verified in official R documentation and widely used in ecosystem
- Pitfalls: HIGH - Identified by code inspection of current implementation, specific to this codebase's bugs

**Research date:** 2026-01-22
**Valid until:** 60 days (stable domain - base R features unlikely to change, hdf5r is mature)
