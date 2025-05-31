#' Validation Functions for fmri_dataset Objects
#'
#' This file implements validation functions to ensure internal consistency
#' and data integrity of `fmri_dataset` objects. Provides comprehensive checks
#' across all dataset types and configurations.
#'
#' @name fmri_dataset_validate
NULL

#' Validate fmri_dataset Object
#'
#' **Ticket #19**: Comprehensive validation function that checks internal consistency
#' of an `fmri_dataset` object. Verifies data integrity, dimensional compatibility,
#' and temporal bounds across all components.
#'
#' @param x An `fmri_dataset` object
#' @param check_data_load Logical indicating whether to validate by loading actual data.
#'   If FALSE (default), performs lightweight checks. If TRUE, loads data to verify
#'   dimensions match metadata (may be slow for large datasets).
#' @param verbose Logical indicating whether to print validation progress (default: FALSE)
#' 
#' @return Logical TRUE if all checks pass, or throws informative error messages
#' 
#' @details
#' **Validation Checks Performed**:
#' \enumerate{
#'   \item \strong{Object Structure}: Validates fmri_dataset class and required fields
#'   \item \strong{Sampling Frame}: Checks sampling_frame object consistency
#'   \item \strong{Data Sources}: Validates image and mask source integrity
#'   \item \strong{Dimensional Consistency}: Verifies run_lengths match data dimensions
#'   \item \strong{Mask Compatibility}: Ensures mask dimensions match image dimensions
#'   \item \strong{Event Table Bounds}: Validates event onsets/durations within time bounds
#'   \item \strong{Censor Vector}: Checks censor_vector length matches total timepoints
#'   \item \strong{Metadata Integrity}: Validates metadata consistency
#' }
#' 
#' **Performance Notes**:
#' - With `check_data_load = FALSE`: Fast validation using metadata only
#' - With `check_data_load = TRUE`: Complete validation including data loading
#' 
#' @examples
#' \dontrun{
#' # Quick validation (metadata only)
#' validate_fmri_dataset(dataset)
#' 
#' # Complete validation (loads data)
#' validate_fmri_dataset(dataset, check_data_load = TRUE, verbose = TRUE)
#' }
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_dataset_create}}, \code{\link{is.fmri_dataset}}
validate_fmri_dataset <- function(x, check_data_load = FALSE, verbose = FALSE) {
  
  if (verbose) cat("🔍 Validating fmri_dataset object...\n")
  
  # === 1. Object Structure Validation ===
  if (verbose) cat("  • Checking object structure...\n")
  validate_object_structure(x)
  
  # === 2. Sampling Frame Validation ===
  if (verbose) cat("  • Validating sampling frame...\n")
  validate_sampling_frame_integrity(x)
  
  # === 3. Data Source Validation ===
  if (verbose) cat("  • Validating data sources...\n")
  validate_data_sources(x)
  
  # === 4. Dimensional Consistency (metadata-based) ===
  if (verbose) cat("  • Checking dimensional consistency...\n")
  validate_dimensional_consistency(x, check_data_load)
  
  # === 5. Mask Compatibility ===
  if (verbose) cat("  • Validating mask compatibility...\n")
  validate_mask_compatibility(x, check_data_load)
  
  # === 6. Event Table Validation ===
  if (verbose) cat("  • Validating event table...\n")
  validate_event_table_bounds(x)
  
  # === 7. Censor Vector Validation ===
  if (verbose) cat("  • Validating censor vector...\n")
  validate_censor_vector_consistency(x)
  
  # === 8. Metadata Integrity ===
  if (verbose) cat("  • Checking metadata integrity...\n")
  validate_metadata_integrity(x)
  
  if (verbose) cat("✅ All validation checks passed!\n")
  return(TRUE)
}

#' Check if Object is fmri_dataset
#'
#' @param x Object to test
#' @return Logical indicating if x is an fmri_dataset
#' @export
#' @family fmri_dataset
is.fmri_dataset <- function(x) {
  inherits(x, "fmri_dataset")
}

# ============================================================================
# Internal Validation Functions
# ============================================================================

#' Validate Object Structure
#' 
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
validate_object_structure <- function(x) {
  
  if (!is.fmri_dataset(x)) {
    stop("Object is not of class 'fmri_dataset'")
  }
  
  # Check required top-level fields
  required_fields <- c("sampling_frame", "metadata", "data_cache")
  missing_fields <- setdiff(required_fields, names(x))
  if (length(missing_fields) > 0) {
    stop("Missing required fields: ", paste(missing_fields, collapse = ", "))
  }
  
  # Check metadata structure
  if (!is.list(x$metadata)) {
    stop("metadata must be a list")
  }
  
  # Check that data_cache is an environment
  if (!is.environment(x$data_cache)) {
    stop("data_cache must be an environment")
  }
  
  # Check that exactly one data source is populated
  data_sources <- c("image_paths", "image_objects", "image_matrix")
  populated_sources <- sapply(data_sources, function(field) !is.null(x[[field]]))
  
  if (sum(populated_sources) == 0) {
    stop("No image data source found. One of image_paths, image_objects, or image_matrix must be populated.")
  }
  
  if (sum(populated_sources) > 1) {
    stop("Multiple image data sources found. Only one of image_paths, image_objects, or image_matrix should be populated.")
  }
}

#' Validate Sampling Frame Integrity
#' 
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
validate_sampling_frame_integrity <- function(x) {
  
  sf <- x$sampling_frame
  
  if (is.null(sf)) {
    stop("sampling_frame is NULL")
  }
  
  if (!is.sampling_frame(sf)) {
    stop("sampling_frame is not of class 'sampling_frame'")
  }
  
  # Validate sampling frame internal consistency
  if (sf$total_timepoints != sum(sf$blocklens)) {
    stop("sampling_frame internal inconsistency: total_timepoints (", sf$total_timepoints, 
         ") does not equal sum of blocklens (", sum(sf$blocklens), ")")
  }
  
  if (sf$n_runs != length(sf$blocklens)) {
    stop("sampling_frame internal inconsistency: n_runs (", sf$n_runs, 
         ") does not equal length of blocklens (", length(sf$blocklens), ")")
  }
  
  # Check TR is positive
  if (any(sf$TR <= 0)) {
    stop("TR values must be positive, found: ", paste(sf$TR[sf$TR <= 0], collapse = ", "))
  }
  
  # Check run lengths are positive integers
  if (any(sf$blocklens <= 0)) {
    stop("Run lengths must be positive, found: ", paste(sf$blocklens[sf$blocklens <= 0], collapse = ", "))
  }
}

#' Validate Data Sources
#' 
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
validate_data_sources <- function(x) {
  
  dataset_type <- x$metadata$dataset_type
  
  if (is.null(dataset_type)) {
    stop("metadata$dataset_type is NULL")
  }
  
  valid_types <- VALID_DATASET_TYPES
  if (!dataset_type %in% valid_types) {
    stop("Invalid dataset_type: ", dataset_type, ". Must be one of: ", paste(valid_types, collapse = ", "))
  }
  
  # Validate data source consistency with dataset_type
  if (dataset_type %in% c("file_vec", "bids_file")) {
    
    if (is.null(x$image_paths)) {
      stop("dataset_type '", dataset_type, "' requires image_paths to be populated")
    }
    
    if (!is.character(x$image_paths)) {
      stop("image_paths must be character vector for dataset_type '", dataset_type, "'")
    }
    
    # Check files exist
    missing_files <- x$image_paths[!file.exists(x$image_paths)]
    if (length(missing_files) > 0) {
      stop("Image files not found: ", paste(missing_files, collapse = ", "))
    }
    
  } else if (dataset_type %in% c("memory_vec", "bids_mem")) {
    
    if (is.null(x$image_objects)) {
      stop("dataset_type '", dataset_type, "' requires image_objects to be populated")
    }
    
    if (!is.list(x$image_objects)) {
      stop("image_objects must be a list for dataset_type '", dataset_type, "'")
    }
    
  } else if (dataset_type == "matrix") {
    
    if (is.null(x$image_matrix)) {
      stop("dataset_type 'matrix' requires image_matrix to be populated")
    }
    
    if (!is.matrix(x$image_matrix) && !is.array(x$image_matrix)) {
      stop("image_matrix must be a matrix or array for dataset_type 'matrix'")
    }
  }
  
  # Validate mask source if present
  mask_sources <- c("mask_path", "mask_object", "mask_vector")
  populated_masks <- sapply(mask_sources, function(field) !is.null(x[[field]]))
  
  if (sum(populated_masks) > 1) {
    stop("Multiple mask sources found. Only one of mask_path, mask_object, or mask_vector should be populated.")
  }
  
  # Check mask file exists if mask_path is used
  if (!is.null(x$mask_path)) {
    if (!is.character(x$mask_path) || length(x$mask_path) != 1) {
      stop("mask_path must be a single character string")
    }
    if (!file.exists(x$mask_path)) {
      stop("Mask file not found: ", x$mask_path)
    }
  }
}

#' Validate Dimensional Consistency
#' 
#' @param x fmri_dataset object
#' @param check_data_load Whether to load data for validation
#' @keywords internal
#' @noRd
validate_dimensional_consistency <- function(x, check_data_load) {
  
  sf <- x$sampling_frame
  expected_timepoints <- sf$total_timepoints
  
  # If censoring is applied, adjust expected timepoints
  if (!is.null(x$censor_vector)) {
    # Count non-censored timepoints (TRUE means keep, FALSE means censor)
    if (is.logical(x$censor_vector)) {
      expected_timepoints <- sum(x$censor_vector)
    } else if (is.numeric(x$censor_vector)) {
      expected_timepoints <- sum(x$censor_vector == 1)
    }
  }
  
  if (check_data_load) {
    # Load data and check actual dimensions
    tryCatch({
      data_matrix <- get_data_matrix(x, apply_transformations = FALSE)
      actual_timepoints <- nrow(data_matrix)
      
      if (actual_timepoints != expected_timepoints) {
        stop("Data dimension mismatch: expected ", expected_timepoints, 
             " timepoints (after censoring), but loaded data has ", actual_timepoints, " timepoints")
      }
      
    }, error = function(e) {
      stop("Error loading data for validation: ", e$message)
    })
    
  } else {
    # Lightweight check for matrix data only
    if (x$metadata$dataset_type == "matrix") {
      actual_timepoints <- nrow(x$image_matrix)
      # For matrix data, we check against the original uncensored timepoints
      # since censoring is applied during get_data_matrix, not stored in the matrix
      original_expected <- sf$total_timepoints
      if (actual_timepoints != original_expected) {
        stop("Matrix dimension mismatch: sampling_frame indicates ", original_expected, 
             " timepoints, but image_matrix has ", actual_timepoints, " timepoints")
      }
    }
  }
}

#' Validate Mask Compatibility
#' 
#' @param x fmri_dataset object
#' @param check_data_load Whether to load data for validation
#' @keywords internal
#' @noRd
validate_mask_compatibility <- function(x, check_data_load) {
  
  # Only validate if mask is present
  if (is.null(x$mask_path) && is.null(x$mask_object) && is.null(x$mask_vector)) {
    return(TRUE)
  }
  
  if (check_data_load) {
    # Load mask and data to check compatibility
    tryCatch({
      mask <- get_mask_volume(x, as_vector = TRUE)
      if (is.null(mask)) return(TRUE)
      
      # For matrix data, check directly
      if (x$metadata$dataset_type == "matrix") {
        n_voxels_data <- ncol(x$image_matrix)
        n_voxels_mask <- length(mask)
        
        if (n_voxels_data != n_voxels_mask) {
          stop("Mask compatibility error: image_matrix has ", n_voxels_data, 
               " voxels, but mask has ", n_voxels_mask, " voxels")
        }
      }
      
      # For other types, we'd need to load data (expensive)
      # This is handled by get_data_matrix which applies the mask
      
    }, error = function(e) {
      stop("Error validating mask compatibility: ", e$message)
    })
    
  } else {
    # Lightweight check for matrix + mask_vector only
    if (x$metadata$dataset_type == "matrix" && !is.null(x$mask_vector)) {
      n_voxels_data <- ncol(x$image_matrix)
      n_voxels_mask <- length(x$mask_vector)
      
      if (n_voxels_data != n_voxels_mask) {
        stop("Mask compatibility error: image_matrix has ", n_voxels_data, 
             " voxels, but mask_vector has ", n_voxels_mask, " voxels")
      }
    }
  }
}

#' Validate Event Table Bounds
#' 
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
validate_event_table_bounds <- function(x) {
  
  event_table <- x$event_table
  
  if (is.null(event_table) || nrow(event_table) == 0) {
    return(TRUE)
  }
  
  if (!is.data.frame(event_table)) {
    stop("event_table must be a data.frame or tibble")
  }
  
  sf <- x$sampling_frame
  total_duration <- get_total_duration(sf)
  
  # Check for onset column
  if ("onset" %in% names(event_table)) {
    onsets <- event_table$onset
    
    if (any(is.na(onsets))) {
      stop("event_table contains NA values in onset column")
    }
    
    if (any(onsets < 0)) {
      stop("event_table contains negative onset values: ", paste(onsets[onsets < 0], collapse = ", "))
    }
    
    if (any(onsets > total_duration)) {
      stop("event_table contains onset values beyond total duration (", total_duration, " sec): ", 
           paste(onsets[onsets > total_duration], collapse = ", "))
    }
  }
  
  # Check for duration column if present
  if ("duration" %in% names(event_table)) {
    durations <- event_table$duration
    
    if (any(is.na(durations))) {
      stop("event_table contains NA values in duration column")
    }
    
    if (any(durations < 0)) {
      stop("event_table contains negative duration values: ", paste(durations[durations < 0], collapse = ", "))
    }
    
    # Check that onset + duration doesn't exceed total duration
    if ("onset" %in% names(event_table)) {
      end_times <- event_table$onset + durations
      if (any(end_times > total_duration)) {
        stop("event_table contains events extending beyond total duration (", total_duration, " sec)")
      }
    }
  }
}

#' Validate Censor Vector Consistency
#' 
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
validate_censor_vector_consistency <- function(x) {
  
  censor_vector <- x$censor_vector
  
  if (is.null(censor_vector)) {
    return(TRUE)
  }
  
  if (!is.logical(censor_vector) && !is.numeric(censor_vector)) {
    stop("censor_vector must be logical or numeric")
  }
  
  sf <- x$sampling_frame
  expected_length <- sf$total_timepoints
  actual_length <- length(censor_vector)
  
  if (actual_length != expected_length) {
    stop("censor_vector length (", actual_length, ") does not match total timepoints (", 
         expected_length, ") from sampling_frame")
  }
  
  # Convert to logical if numeric and check values
  if (is.numeric(censor_vector)) {
    if (!all(censor_vector %in% c(0, 1))) {
      stop("Numeric censor_vector must contain only 0 and 1 values")
    }
  }
}

#' Validate Metadata Integrity
#' 
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
validate_metadata_integrity <- function(x) {
  
  metadata <- x$metadata
  
  # Check required metadata fields
  if (is.null(metadata$dataset_type)) {
    stop("metadata$dataset_type is required")
  }
  
  # Validate TR consistency between metadata and sampling_frame
  if (!is.null(metadata$TR)) {
    sf_TR <- x$sampling_frame$TR[1]  # Get first TR value
    if (!isTRUE(all.equal(metadata$TR, sf_TR, tolerance = 1e-6))) {
      stop("metadata$TR (", metadata$TR, ") does not match sampling_frame$TR (", sf_TR, ")")
    }
  }
  
  # Validate file_options if present
  if (!is.null(metadata$file_options)) {
    if (!is.list(metadata$file_options)) {
      stop("metadata$file_options must be a list")
    }
    
    valid_modes <- c("normal", "bigvec", "mmap", "filebacked")
    if (!is.null(metadata$file_options$mode)) {
      if (!metadata$file_options$mode %in% valid_modes) {
        stop("Invalid file_options$mode: ", metadata$file_options$mode, 
             ". Must be one of: ", paste(valid_modes, collapse = ", "))
      }
    }
  }
  
  # Validate matrix_options if present
  if (!is.null(metadata$matrix_options)) {
    if (!is.list(metadata$matrix_options)) {
      stop("metadata$matrix_options must be a list")
    }
    
    logical_options <- c("temporal_zscore", "voxelwise_detrend")
    for (opt in logical_options) {
      if (!is.null(metadata$matrix_options[[opt]])) {
        if (!is.logical(metadata$matrix_options[[opt]])) {
          stop("metadata$matrix_options$", opt, " must be logical")
        }
      }
    }
  }
  
  # Validate BIDS info if present
  if (!is.null(metadata$bids_info)) {
    if (!is.list(metadata$bids_info)) {
      stop("metadata$bids_info must be a list")
    }
  }
} 