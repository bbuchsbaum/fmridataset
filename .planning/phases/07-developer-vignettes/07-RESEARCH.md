# Phase 7: Developer Vignettes - Research

**Researched:** 2026-01-23
**Domain:** R package vignette development, storage backend architecture, developer documentation
**Confidence:** HIGH

## Summary

This research examined the current state of three developer vignettes (backend-development-basics.Rmd, backend-registry.Rmd, extending-backends.Rmd) and the actual backend API implementation in fmridataset. The vignettes contain comprehensive content but use `eval=FALSE` for most code blocks, indicating examples may not be executable or tested against the current API.

The backend contract requires six S3 methods: `backend_open`, `backend_close`, `backend_get_dims`, `backend_get_mask`, `backend_get_data`, and `backend_get_metadata`. The registry system provides `register_backend()`, `create_backend()`, `is_backend_registered()`, and related functions for pluggable backend support.

**Key findings:**
- Vignettes already contain detailed, well-structured content with realistic examples
- Most code blocks use `eval=FALSE`, preventing automatic testing during vignette build
- Backend contract is well-defined in R/storage_backend.R with validation function
- Registry system is fully implemented with test coverage
- Examples in vignettes appear API-compatible but need verification through execution

**Primary recommendation:** Enable `eval=TRUE` for code blocks with synthetic data, add setup code to make examples self-contained, verify all API calls match current implementation, and ensure examples execute without errors.

## Standard Stack

The vignettes demonstrate R package documentation patterns and backend development:

### Core
| Library | Purpose | Why Standard |
|---------|---------|--------------|
| knitr | Vignette rendering | Standard R package vignette engine |
| rmarkdown | Markdown processing | Standard for R package long-form docs |
| testthat patterns | Example validation | Referenced for testing concepts |

### Supporting (Mentioned in Examples)
| Library | Purpose | When to Use |
|---------|---------|-------------|
| jsonlite | JSON backend examples | When demonstrating custom format backends |
| digest | Checksum validation | For data integrity examples in extending-backends |
| microbenchmark | Performance examples | Optional, for performance optimization sections |
| profvis/profmem | Profiling examples | Optional, for memory management demonstrations |

**Installation:**
```r
# Core vignette dependencies (already in DESCRIPTION)
# Supporting packages for optional examples
install.packages(c("jsonlite", "digest", "microbenchmark"))
```

## Architecture Patterns

### Vignette Organization
The three vignettes follow a progressive learning path:

```
Developer Vignettes/
├── backend-development-basics.Rmd  # Entry point - minimal viable backend
├── backend-registry.Rmd            # Registry usage and integration
└── extending-backends.Rmd          # Advanced patterns and production features
```

### Pattern 1: Incremental Backend Construction
**What:** Build from minimal viable backend to full implementation
**When to use:** Teaching backend development fundamentals
**Example structure:**
```r
# Step 1: Constructor with validation
custom_backend <- function(source, ...) {
  backend <- list(source = source, ...)
  class(backend) <- c("custom_backend", "storage_backend")
  backend
}

# Step 2: Required methods
backend_open.custom_backend <- function(backend) { ... }
backend_get_dims.custom_backend <- function(backend) { ... }
# etc.
```
**Source:** All three vignettes use this pattern

### Pattern 2: Synthetic Data for Self-Contained Examples
**What:** Create in-memory test data instead of requiring external files
**When to use:** Vignette examples that need to be executable
**Example:**
```r
# From backend-development-basics.Rmd lines 167-175
set.seed(123)
n_time <- 100
n_voxels <- 500
backend$data_cache <- matrix(
  rnorm(n_time * n_voxels),
  nrow = n_time,
  ncol = n_voxels
)
```
**Source:** Used throughout backend-development-basics.Rmd

### Pattern 3: State Management Lifecycle
**What:** Demonstrate open/close lifecycle with resource tracking
**When to use:** All backend implementations
**Example:**
```r
# From stateful_backend example (lines 334-386)
backend_open.custom_backend <- function(backend) {
  if (backend$is_open) return(backend)  # Idempotent
  # Acquire resources
  backend$is_open <- TRUE
  backend
}
```

### Anti-Patterns Documented

**From vignettes:**
- **Eager loading:** Don't load all data in constructor (basics line 799-801)
- **Missing validation:** Always check `is_open` state before operations (basics line 938-941)
- **Non-idempotent operations:** `backend_open()` must be safe to call multiple times (basics line 808-810)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backend validation | Custom validation checks | `validate_backend()` in R/storage_backend.R | Comprehensive validation with proper error messages, checks class inheritance, method existence, dimension format |
| Registry management | Custom backend lookup | `register_backend()`, `create_backend()` | Thread-safe registry, validation, error handling, introspection |
| Temporary file creation | Manual tempfile paths | `tempfile()` with cleanup | Automatic cleanup, cross-platform compatibility |

**Key insight:** The backend contract validation (lines 118-204 in R/storage_backend.R) handles all standard validation. Custom backends only need to implement the six required methods; the framework handles method existence checking, dimension format validation, and mask consistency.

## Common Pitfalls

### Pitfall 1: Using `eval=FALSE` Without Testing
**What goes wrong:** Code blocks marked `eval=FALSE` can become outdated when APIs change
**Why it happens:** Developers use `eval=FALSE` to avoid execution errors during development
**How to avoid:**
- Use `eval=TRUE` with proper setup code
- Add `error=TRUE` for examples demonstrating error handling
- Use synthetic data to avoid external dependencies
**Warning signs:** Functions that don't exist, incorrect parameter names, outdated return values

**Evidence:** backend-registry.Rmd line 20 sets `eval=FALSE` globally; extending-backends.Rmd line 20 sets `eval=FALSE` globally

### Pitfall 2: Missing Required S3 Method Exports
**What goes wrong:** Backend methods aren't found by S3 dispatch
**Why it happens:** Methods defined in vignettes or tests don't get registered
**How to avoid:** Document that production backends need proper S3 method exports or use `assign()` pattern for examples
**Warning signs:** "no applicable method" errors during dispatch

**Evidence:** test_backend_registry.R lines 72-85 show the pattern of assigning S3 methods to globalenv() for testing

### Pitfall 3: Mask vs. Masked Data Confusion
**What goes wrong:** Confusion between `backend_get_data()` applying mask vs. returning all spatial data
**Why it happens:** Different backends handle masking at different layers
**How to avoid:** Document that `backend_get_data()` should return only masked voxels (cols parameter indexes into masked data, not full spatial grid)
**Warning signs:** Dimension mismatches between mask length and data columns

**Evidence:** matrix_backend.R lines 134-148 show correct pattern: apply mask first, then apply col subsetting to masked data

### Pitfall 4: Vignette Examples Without Setup Dependencies
**What goes wrong:** Examples reference objects or functions not defined in the vignette
**Why it happens:** Copy-paste from other code without self-contained setup
**How to avoid:** Each major example should have complete setup code
**Warning signs:** Undefined variable errors, missing library() calls

**Evidence:** backend-development-basics.Rmd line 29-32 sources helper file that may not exist during vignette build

## Code Examples

Verified patterns from official sources:

### Minimal Backend Constructor
```r
# Source: R/matrix_backend.R lines 20-96
matrix_backend <- function(data_matrix, mask = NULL, spatial_dims = NULL, metadata = NULL) {
  # Input validation
  if (!is.matrix(data_matrix)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "data_matrix must be a matrix",
      parameter = "data_matrix",
      value = class(data_matrix)
    )
  }

  # Default values
  if (is.null(mask)) {
    mask <- rep(TRUE, ncol(data_matrix))
  }

  # Create backend object
  backend <- list(
    data_matrix = data_matrix,
    mask = mask,
    spatial_dims = spatial_dims %||% c(ncol(data_matrix), 1, 1),
    metadata = metadata %||% list()
  )

  class(backend) <- c("matrix_backend", "storage_backend")
  backend
}
```

### Backend Registration
```r
# Source: R/backend_registry.R lines 66-118
register_backend <- function(name, factory, description = NULL,
                             validate_function = NULL, overwrite = FALSE) {
  # Validate inputs
  if (!is.character(name) || length(name) != 1 || nchar(name) == 0) {
    stop_fmridataset(fmridataset_error_config, "name must be a non-empty character string")
  }

  if (!is.function(factory)) {
    stop_fmridataset(fmridataset_error_config, "factory must be a function")
  }

  # Check for existing registration
  if (exists(name, envir = .backend_registry) && !overwrite) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Backend '%s' is already registered. Use overwrite = TRUE to replace.", name)
    )
  }

  # Create and store registration
  registration <- list(
    name = name,
    factory = factory,
    description = description %||% paste("Backend:", name),
    validate_function = validate_function,
    registered_at = Sys.time()
  )

  assign(name, registration, envir = .backend_registry)
  invisible(TRUE)
}
```

### Backend Data Access Pattern
```r
# Source: R/matrix_backend.R lines 134-148
backend_get_data.matrix_backend <- function(backend, rows = NULL, cols = NULL) {
  # First, apply the mask to get the matrix of valid voxels
  data <- backend$data_matrix[, backend$mask, drop = FALSE]

  # Apply row subsetting
  if (!is.null(rows)) {
    data <- data[rows, , drop = FALSE]
  }

  # Apply column subsetting (to masked data)
  if (!is.null(cols)) {
    data <- data[, cols, drop = FALSE]
  }

  data
}
```

## State of the Art

| Aspect | Current State | Notes |
|--------|---------------|-------|
| Backend contract | Stable - 6 required methods | Defined in R/storage_backend.R, unlikely to change |
| Registry system | Full implementation | Complete with validation, introspection, error handling |
| Vignette content | Comprehensive but untested | Good explanations, but `eval=FALSE` prevents verification |
| Example backends | Multiple working implementations | matrix_backend, nifti_backend, h5_backend, zarr_backend, study_backend |
| Test coverage | Strong for actual backends | test_backend_registry.R shows pattern, actual backends have test files |

**Recent changes (from git log):**
- Significant test coverage work in phase 4 (commits 172e6d6 through 114c653)
- H5 backend vignette made executable in phase 6 (commit de1b0d8)
- zarr_backend tests added (commit cdb640a)

**Current approach vs older patterns:**
- **Old:** Backends implemented ad-hoc without registry
- **Current:** Pluggable registry system with validation
- **Impact:** External packages can add backends without modifying fmridataset

## Open Questions

Items requiring validation during implementation:

1. **Helper file dependency**
   - What we know: backend-development-basics.Rmd line 29-32 sources "../R/vignette_helpers.R"
   - What's unclear: Does this file exist? What functions does it provide?
   - Recommendation: Check if file exists, if not, remove dependency or create minimal helpers

2. **Optional package availability**
   - What we know: extending-backends examples use jsonlite, digest, microbenchmark, profvis, profmem
   - What's unclear: Should vignette gracefully handle missing packages?
   - Recommendation: Use `requireNamespace()` checks and skip examples if packages unavailable

3. **Synthetic data consistency**
   - What we know: Examples use `set.seed()` for reproducibility
   - What's unclear: Should all examples share same seed for consistency?
   - Recommendation: Use consistent seed (123) across all vignette examples

4. **Cross-vignette dependencies**
   - What we know: Vignettes reference each other via links
   - What's unclear: Do examples build on each other or stand alone?
   - Recommendation: Make each vignette self-contained with clear "prerequisites" section

## Sources

### Primary (HIGH confidence)
- R/storage_backend.R - Backend contract specification (lines 1-205)
- R/backend_registry.R - Registry implementation (lines 1-432)
- R/matrix_backend.R - Reference backend implementation (lines 1-159)
- tests/testthat/test_backend_registry.R - Registry usage patterns (lines 1-150)

### Secondary (MEDIUM confidence)
- vignettes/backend-development-basics.Rmd - Current vignette content (lines 1-965)
- vignettes/backend-registry.Rmd - Current vignette content (lines 1-1214)
- vignettes/extending-backends.Rmd - Current vignette content (lines 1-1753)

### Tertiary (Informational)
- Git commit history - Recent backend-related changes
- .planning/phases/07-developer-vignettes/07-CONTEXT.md - User decisions and constraints

## Metadata

**Confidence breakdown:**
- Backend contract: HIGH - Stable API, well-documented, tested
- Registry system: HIGH - Full implementation with tests
- Vignette content accuracy: MEDIUM - Good content but `eval=FALSE` means untested
- API compatibility: MEDIUM - Visual inspection suggests compatible but needs execution verification

**Research date:** 2026-01-23
**Valid until:** 60 days (backend API is stable, unlikely to change rapidly)

**Key validation needs:**
1. Execute all code blocks to verify no API mismatches
2. Check that helper file dependencies are resolved
3. Verify synthetic data examples produce expected output
4. Confirm error handling examples demonstrate correct error types
