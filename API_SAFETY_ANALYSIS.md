# fmridataset API Safety and Usability Analysis

This document analyzes the fmridataset package for API safety and
usability issues, providing specific examples and backward-compatible
improvement suggestions.

## 1. Type Safety Issues

### 1.1 Multiple Incompatible Type Acceptance Without Clear Validation

**Issue**: Several functions accept multiple types but don’t validate or
provide clear error messages:

``` r
# In dataset_constructors.R
matrix_dataset <- function(datamat, TR, run_length, event_table = data.frame()) {
  if (is.vector(datamat)) {
    datamat <- as.matrix(datamat)  # Silent coercion
  }
  assert_that(sum(run_length) == nrow(datamat))
  # ...
}
```

**Problems**: - Silently coerces vectors to matrices without warning -
No validation that the coercion produces expected dimensions -
`run_length` can be a single value or vector, but no clear documentation

**Suggested Improvement**:

``` r
matrix_dataset <- function(datamat, TR, run_length, event_table = data.frame()) {
  # Explicit type checking with informative messages
  if (is.vector(datamat)) {
    message("Converting vector to single-column matrix")
    datamat <- as.matrix(datamat)
  }
  
  if (!is.matrix(datamat)) {
    stop_fmridataset(
      fmridataset_error_config,
      "datamat must be a matrix or vector",
      parameter = "datamat",
      value = class(datamat)
    )
  }
  
  # Validate dimensions match
  if (!is.numeric(run_length) || any(run_length <= 0)) {
    stop_fmridataset(
      fmridataset_error_config,
      "run_length must be positive numeric values",
      parameter = "run_length",
      value = run_length
    )
  }
  
  # Rest of function...
}
```

### 1.2 Inconsistent Return Types

**Issue**:
[`get_mask()`](https://bbuchsbaum.github.io/fmridataset/reference/get_mask.md)
returns different types depending on dataset type:

``` r
# In data_access.R
get_mask.fmri_file_dataset <- function(x, ...) {
  if (!is.null(x$backend)) {
    mask_vec <- backend_get_mask(x$backend)  # Returns logical vector
    dims <- backend_get_dims(x$backend)$spatial
    array(mask_vec, dims)  # Converts to 3D array
  } else {
    # Legacy path returns NeuroVol object
    neuroim2::read_vol(x$mask_file)
  }
}

get_mask.matrix_dataset <- function(x, ...) {
  x$mask  # Returns numeric vector (rep(1, ncol))
}

get_mask.latent_dataset <- function(x, ...) {
  x$lvec@mask  # Returns logical vector from S4 slot
}
```

**Problems**: - Returns 3D array, NeuroVol, numeric vector, or logical
vector - Users can’t predict return type without knowing dataset
internals - Type inconsistency makes generic programming difficult

**Suggested Improvement**: Add a `standardize` parameter:

``` r
get_mask <- function(x, standardize = TRUE, ...) {
  UseMethod("get_mask")
}

get_mask.fmri_file_dataset <- function(x, standardize = TRUE, ...) {
  raw_mask <- # ... existing code
  
  if (standardize) {
    # Always return logical vector for consistency
    return(as.logical(as.vector(raw_mask)))
  }
  return(raw_mask)  # Return native format if requested
}
```

## 2. Error Message Issues

### 2.1 Cryptic Error Messages

**Issue**: Many errors don’t guide users to solutions:

``` r
# In dataset_constructors.R
assert_that(sum(run_length) == nrow(datamat))
# Produces: "sum(run_length) == nrow(datamat) is not TRUE"
```

**Problems**: - Doesn’t tell user what the actual values were - Doesn’t
suggest how to fix the issue - No context about which parameter is wrong

**Suggested Improvement**:

``` r
if (sum(run_length) != nrow(datamat)) {
  stop_fmridataset(
    fmridataset_error_config,
    sprintf(
      "Total run length (%d) must equal number of timepoints in data (%d). ",
      "Check that run_length values sum to your data's time dimension.",
      sum(run_length), nrow(datamat)
    ),
    parameter = "run_length",
    value = run_length,
    expected = nrow(datamat)
  )
}
```

### 2.2 Missing Input Validation Leading to Downstream Errors

**Issue**:
[`fmri_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_dataset.md)
doesn’t validate file existence early:

``` r
# Current code allows creation with non-existent files
# Error occurs later during data access
backend <- nifti_backend(
  source = scan_files,  # No validation here
  mask_source = maskfile,
  preload = preload,
  mode = mode
)
```

**Suggested Improvement**: Add upfront validation:

``` r
# In fmri_dataset()
if (!all(file.exists(scan_files))) {
  missing <- scan_files[!file.exists(scan_files)]
  stop_fmridataset(
    fmridataset_error_backend_io,
    sprintf(
      "Cannot find scan files: %s\n",
      "Please check file paths and ensure files exist.",
      paste(missing, collapse = ", ")
    ),
    file = missing,
    operation = "validate"
  )
}
```

## 3. API Consistency Issues

### 3.1 Inconsistent Parameter Names

**Issue**: Similar concepts use different parameter names:

``` r
# Different names for mask across functions:
fmri_dataset(scans, mask = NULL, ...)           # "mask" 
fmri_mem_dataset(scans, mask, ...)              # "mask" (required)
fmri_h5_dataset(h5_files, mask_source, ...)     # "mask_source"
latent_backend(source, mask_source = NULL, ...)  # "mask_source"
```

**Problems**: - Users must remember different parameter names for same
concept - Can’t easily switch between dataset types - Increases
cognitive load

**Suggested Improvement**: Add aliases for backward compatibility:

``` r
fmri_dataset <- function(scans, mask = NULL, mask_source = NULL, ...) {
  # Support both parameter names
  if (!is.null(mask_source) && is.null(mask)) {
    mask <- mask_source
  }
  # Rest of function...
}
```

### 3.2 Confusing Function Names

**Issue**: Function names don’t clearly indicate their purpose:

``` r
# Unclear what these return:
get_data()        # Returns native format (matrix, NeuroVec, etc.)
get_data_matrix() # Always returns matrix
samples()         # Returns 1:n_timepoints (just indices)
blocklens()       # Returns run lengths
```

**Suggested Improvement**: Add clearer aliases:

``` r
# Add more descriptive aliases
get_data_native <- get_data
get_timepoint_indices <- samples
get_run_lengths <- blocklens  # This already exists but isn't prominent
```

## 4. Usability Issues

### 4.1 Common Tasks Require Too Many Steps

**Issue**: Getting a simple data matrix requires multiple operations:

``` r
# Current approach for masked data extraction:
dset <- fmri_dataset(scans, mask, TR, run_length)
data <- get_data(dset)  # Returns NeuroVec
mask <- get_mask(dset)  # Returns NeuroVol
masked_data <- neuroim2::series(data, which(mask != 0))
```

**Suggested Improvement**: Add convenience function:

``` r
# Add to API:
get_masked_matrix <- function(x, ...) {
  UseMethod("get_masked_matrix")
}

get_masked_matrix.fmri_dataset <- function(x, ...) {
  # Direct path to masked matrix
  if (!is.null(x$backend)) {
    return(backend_get_data(x$backend, ...))
  }
  # Legacy path
  get_data_matrix(x, ...)
}
```

### 4.2 Unclear Defaults

**Issue**: `matrix_dataset` creates a default mask of all 1s without
documentation:

``` r
matrix_dataset <- function(datamat, TR, run_length, event_table = data.frame()) {
  # ...
  ret <- list(
    # ...
    mask = rep(1, ncol(datamat))  # Undocumented default
  )
}
```

**Problems**: - Users might not realize all voxels are included by
default - Numeric 1s instead of logical TRUE is inconsistent

**Suggested Improvement**:

``` r
matrix_dataset <- function(datamat, TR, run_length, 
                          event_table = data.frame(),
                          mask = NULL) {
  # ...
  if (is.null(mask)) {
    message("Using default mask (all voxels included)")
    mask <- rep(TRUE, ncol(datamat))
  }
  # Validate mask
  if (length(mask) != ncol(datamat)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Mask length (%d) must match number of voxels (%d)",
              length(mask), ncol(datamat)),
      parameter = "mask",
      value = length(mask)
    )
  }
}
```

### 4.3 Required Parameters That Could Have Sensible Defaults

**Issue**: TR is always required even when not needed:

``` r
matrix_dataset <- function(datamat, TR, run_length, event_table = data.frame()) {
  # TR is required but might not be used in some analyses
}
```

**Suggested Improvement**: Make TR optional with warning:

``` r
matrix_dataset <- function(datamat, TR = NULL, run_length, 
                          event_table = data.frame()) {
  if (is.null(TR)) {
    warning("No TR specified. Using default of 1.0 second. ",
            "Specify TR explicitly if timing matters for your analysis.")
    TR <- 1.0
  }
  # Rest of function...
}
```

## 5. Safety Issues

### 5.1 Missing Resource Cleanup

**Issue**: Backend resources aren’t automatically cleaned up:

``` r
# No automatic cleanup in dataset destructors
# File handles or memory maps might leak
backend <- backend_open(backend)
# No corresponding close in error paths
```

**Suggested Improvement**: Add finalizers:

``` r
# In dataset constructors:
backend <- backend_open(backend)

# Register cleanup
reg.finalizer(ret, function(obj) {
  if (!is.null(obj$backend) && 
      inherits(obj$backend, "storage_backend")) {
    try(backend_close(obj$backend), silent = TRUE)
  }
}, onexit = TRUE)
```

### 5.2 Side Effects Not Documented

**Issue**: `latent_backend` returns latent scores instead of voxel data,
but this isn’t clear from function names:

``` r
# This returns components, not voxels!
data <- get_data_matrix(latent_dataset_obj)
# Returns: time × components, not time × voxels
```

**Suggested Improvement**: Add explicit documentation and consider
renaming:

``` r
#' @details
#' IMPORTANT: For latent datasets, get_data_matrix() returns the latent
#' scores (time × components), not reconstructed voxel data. Use
#' get_latent_scores() for clarity.

get_latent_scores <- function(x, ...) {
  if (!inherits(x, c("latent_dataset", "fmri_latent_dataset"))) {
    stop("get_latent_scores only works with latent datasets")
  }
  get_data_matrix(x, ...)
}
```

## 6. Additional Recommendations

### 6.1 Add Input Validation Helper

Create a centralized validation function:

``` r
validate_fmri_inputs <- function(TR = NULL, run_length = NULL, 
                                mask = NULL, data_dims = NULL) {
  errors <- list()
  
  if (!is.null(TR)) {
    if (!is.numeric(TR) || length(TR) != 1 || TR <= 0) {
      errors$TR <- "TR must be a single positive number"
    }
  }
  
  if (!is.null(run_length)) {
    if (!is.numeric(run_length) || any(run_length <= 0)) {
      errors$run_length <- "run_length must contain positive numbers"
    }
  }
  
  if (length(errors) > 0) {
    msg <- paste(names(errors), errors, sep = ": ", collapse = "\n")
    stop_fmridataset(
      fmridataset_error_config,
      paste("Invalid parameters:\n", msg)
    )
  }
}
```

### 6.2 Add Type Checking Utilities

``` r
ensure_matrix <- function(x, name = "data") {
  if (is.data.frame(x)) {
    message(sprintf("Converting %s from data.frame to matrix", name))
    x <- as.matrix(x)
  } else if (is.vector(x)) {
    message(sprintf("Converting %s from vector to single-column matrix", name))
    x <- as.matrix(x)
  } else if (!is.matrix(x)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("%s must be a matrix, data.frame, or vector", name),
      parameter = name,
      value = class(x)
    )
  }
  x
}
```

### 6.3 Improve Error Context

Add a context parameter to error functions:

``` r
with_fmri_context <- function(expr, context) {
  tryCatch(expr, error = function(e) {
    if (inherits(e, "fmridataset_error")) {
      e$context <- context
    }
    stop(e)
  })
}

# Usage:
with_fmri_context({
  backend <- backend_open(backend)
}, context = list(
  operation = "opening backend",
  dataset_type = class(x)[1]
))
```

## Summary

The main safety and usability issues in fmridataset are:

1.  **Type coercion without warning** - Functions silently convert types
2.  **Inconsistent return types** - Same generic returns different types
3.  **Poor error messages** - Don’t guide users to solutions
4.  **Inconsistent naming** - Similar concepts have different names
5.  **Missing conveniences** - Common tasks require multiple steps
6.  **Resource management** - No automatic cleanup of file handles
7.  **Hidden behavior** - Some functions don’t do what names suggest

All suggested improvements maintain backward compatibility while
improving safety and usability through: - Additional parameters with
defaults - Warning messages for implicit behavior  
- Better error messages with context - Convenience functions and
aliases - Automatic resource cleanup - Clearer documentation
