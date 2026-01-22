# Coding Conventions

**Analysis Date:** 2026-01-22

## Naming Patterns

**Files:**
- All lowercase with underscores separating words: `matrix_backend.R`, `data_chunks.R`, `storage_backend.R`
- Special patterns:
  - Generics defined in `all_generic.R` (loaded first alphabetically)
  - Backend implementations: `[type]_backend.R` (e.g., `nifti_backend.R`, `h5_backend.R`, `zarr_backend.R`)
  - Test files follow `test_[component].R` or `test-[type].R` patterns
  - Golden/snapshot tests: `test-golden-[component].R`

**Functions:**
- Snake_case for all function names: `get_data()`, `backend_open()`, `matrix_dataset()`
- Generic functions declared in `R/all_generic.R`: `get_data()`, `get_data_matrix()`, `data_chunks()`, `blocklens()`
- Constructor pattern:
  - Internal: `new_*()` (e.g., `new_fmri_series()`)
  - User-facing: `*()` (e.g., `matrix_dataset()`, `fmri_dataset()`)
- Backend methods follow pattern: `backend_*()` (e.g., `backend_open()`, `backend_get_data()`, `backend_get_mask()`)
- Method implementations use dot notation: `method.class()` (e.g., `get_data.matrix_dataset()`, `data_chunks.fmri_mem_dataset()`)

**Variables:**
- Snake_case throughout: `data_matrix`, `run_length`, `voxel_ind`, `spatial_dims`, `ntotscans`, `n_timepoints`, `n_voxels`
- Single letters only for loop counters: `i`, `v`, `x`
- Underscores for compound names: `missing_files`, `backend_class`, `signal_voxels`, `noise_voxels`
- Avoid abbreviations in variable names (except standard: `n_` prefix for counts, `x` for data, `i` for indices)

**Types/Classes:**
- S3 classes: lowercase with underscores: `fmri_dataset`, `matrix_dataset`, `fmri_mem_dataset`, `storage_backend`, `matrix_backend`, `nifti_backend`
- Inheritance specified in `class()` call: `class(ret) <- c("matrix_dataset", "fmri_dataset", "list")`
- Error classes follow pattern: `fmridataset_error_*` (e.g., `fmridataset_error_backend_io`, `fmridataset_error_config`)

## Code Style

**Formatting:**
- Line length limit: 120 characters (enforced by lintr, see `.lintr` configuration)
- Object name length limit: 40 characters
- Cyclomatic complexity limit: 15
- Indentation: 2 spaces

**Linting:**
Tool: `lintr` via `.lintr` configuration
- `linters_with_defaults()` with custom settings:
  - `line_length_linter(120)` - enforce 120 char lines
  - `object_length_linter(40)` - limit identifiers to 40 chars
  - `cyclocomp_linter(15)` - limit cyclomatic complexity
  - `assignment_linter = NULL` - disabled (allows both `<-` and `=`)
  - `pipe_continuation_linter = NULL` - disabled
- Excluded paths: `tests/testthat.R`, `data-raw/`, `inst/doc/`

**Style/Formatting:**
Applied via GitHub Action workflow (`.github/workflows/style.yaml`)
- Automatic code formatting on commits
- Targets: `R/`, `tests/testthat/`

## Import Organization

**Order (top to bottom):**
1. roxygen2 decorators: `@importFrom`, `@import`, `@keywords`
2. Function body imports/assignments at file level

**Example pattern from `data_access.R`:**
```r
#' @importFrom neuroim2 series
#' @import memoise
```

**Example pattern from `nifti_backend.R`:**
```r
#' @importFrom neuroim2 read_header
#' @importFrom cachem cache_mem
#' @keywords internal
```

**Path Aliases:**
Not explicitly used. Full package imports specified via roxygen decorators or explicit package namespace calls: `neuroim2::series()`, `memoise::memoize()`, `cachem::cache_mem()`.

## Error Handling

**Framework:**
Custom S3 error classes defined in `R/errors.R`:
- `fmridataset_error()` - Base error constructor
- `fmridataset_error_backend_io()` - I/O failures (file not found, read/write errors)
- `fmridataset_error_config()` - Invalid configuration or parameters

**Pattern:**
```r
stop_fmridataset(
  error_fn,  # Constructor: fmridataset_error_config or fmridataset_error_backend_io
  message = "Descriptive message",
  parameter = "parameter_name",  # Optional
  value = actual_value,          # Optional
  ...
)
```

**Examples from codebase:**
```r
# Configuration error
stop_fmridataset(
  fmridataset_error_config,
  message = "data_matrix must be a matrix",
  parameter = "data_matrix",
  value = class(data_matrix)
)

# Backend I/O error
stop_fmridataset(
  fmridataset_error_backend_io,
  message = sprintf("Source files not found: %s", paste(missing_files, collapse = ", ")),
  file = missing_files,
  operation = "open"
)
```

**Validation Pattern:**
Use `assertthat::assert_that()` for preconditions:
```r
assert_that(sum(run_length) == nrow(datamat))
assert_that(all(map_lgl(scans, function(x) inherits(x, "NeuroVec"))))
assert_that(inherits(mask, "NeuroVol"))
```

## Logging

**Framework:** Base R `cat()` for user-facing output

**Patterns:**
- Print methods use `cat()` with ASCII formatting
- Debug/verbose output commented out or in test context
- No formal logging framework (console output only)

**Example from `print_methods.R`:**
```r
cat("\n=== fMRI Dataset ===\n")
cat("Structure: ", paste(class(x), collapse = " > "), "\n", sep = "")
```

## Comments

**When to Comment:**
- At file level: Purpose of module and high-level organization
- Before major sections: Use comment dividers `# ====...====`
- For complex logic: Explain WHY not WHAT
- For temporary workarounds: Mark with context (e.g., "Legacy path for backward compatibility")

**Comment Style:**
- Use `#'` for roxygen documentation (exports, user-facing functions)
- Use `#` for inline/section comments
- Section dividers: `# ===== Section Name =====`

**Example from `fmri_dataset.R` header:**
```r
# ========================================================================
# fMRI Dataset Package - Main Entry Point
# ========================================================================
#
# This file serves as the main entry point for the fmridataset package.
# The original fmri_dataset.R file has been refactored into multiple
# modular files for better maintainability:
...
```

**JSDoc/TSDoc (roxygen2):**
All exported functions documented with roxygen2 tags:
- `@description` - Brief description
- `@param` - Parameter documentation
- `@return` - Return value description
- `@details` - Extended documentation
- `@examples` - Runnable examples (wrapped in `\donttest{}` or `\dontrun{}`)
- `@seealso` - Related functions
- `@export` - Mark for export
- `@keywords internal` - Mark internal/non-exported

## Function Design

**Size:** No explicit limits; typical functions 10-50 lines

**Parameters:**
- Required parameters first
- Optional parameters last with default values
- Use `...` for passing arguments to methods: `function(x, ...) { UseMethod(...) }`

**Return Values:**
- Named lists for complex returns: `list(data = mat, voxel_ind = voxel_ind, row_ind = row_ind, chunk_num = chunk_num)`
- S3 classes constructed with `class(ret) <- c(...)`
- Invisibly return NULL for side-effect functions: `invisible(NULL)`

**Example pattern from `data_chunks.R`:**
```r
data_chunk <- function(mat, voxel_ind, row_ind, chunk_num) {
  ret <- list(
    data = mat,
    voxel_ind = voxel_ind,
    row_ind = row_ind,
    chunk_num = chunk_num
  )
  class(ret) <- c("data_chunk", "list")
  ret
}
```

## Module Design

**Exports:**
- Generic functions exported from `R/all_generic.R`
- Implementation exported from backend files: `@export` tag in roxygen
- Constructor functions exported: `matrix_dataset()`, `fmri_dataset()`, etc.
- Backend constructors exported: `matrix_backend()`, `nifti_backend()`, etc.
- Non-user-facing functions marked: `@keywords internal`

**Barrel Files:**
Not used. Each file focuses on a single concern (one backend per file, one data structure per file).

**File Organization:**
- `R/all_generic.R` - S3 generic declarations (loaded first)
- `R/[component]_backend.R` - Backend implementations
- `R/dataset_constructors.R` - Dataset creators
- `R/data_access.R` - Data retrieval methods
- `R/data_chunks.R` - Chunking and iteration
- `R/print_methods.R` - Display methods
- `R/conversions.R` - Type conversions
- `R/errors.R` - Error classes
- `R/config.R` - Configuration functions

---

*Convention analysis: 2026-01-22*
