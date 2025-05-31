#' Accessor Functions for fmri_dataset Objects
#'
#' This file implements all accessor functions for `fmri_dataset` objects.
#' These functions provide a unified interface for accessing data and metadata
#' regardless of the underlying dataset type (file_vec, memory_vec, matrix, etc.).
#'
#' @name fmri_dataset_accessors
NULL

#' Get the data matrix from an fmri_dataset
#' 
#' Retrieves the full data matrix or subset for specific runs from an fmri_dataset object.
#' Supports applying transformations, masking, and censoring.
#' 
#' @param dataset An fmri_dataset object
#' @param run_id Integer vector specifying which runs to include (default: all runs)
#' @param apply_transformations Logical indicating whether to apply the transformation pipeline (default TRUE)
#' @param verbose Logical indicating whether to print transformation progress (default FALSE)
#' @param ... Additional arguments passed to transformations
#' 
#' @return A matrix with timepoints as rows and voxels as columns
#' 
#' @details 
#' The function applies operations in the following order:
#' \enumerate{
#'   \item Load raw data from source (matrix, files, or objects)
#'   \item Apply spatial masking (if mask is present)
#'   \item Extract specified runs (if run_id specified)
#'   \item Apply transformation pipeline (if apply_transformations = TRUE)
#'   \item Apply temporal censoring (if censor vector is present)
#' }
#' 
#' The transformation pipeline is applied to the data after masking and run selection
#' but before censoring, ensuring that transformations operate on the complete
#' temporal structure.
#' 
#' @examples
#' \dontrun{
#' # Get full data matrix with transformations
#' data_matrix <- get_data_matrix(dataset)
#'
#' # Get raw data without transformations
#' raw_data <- get_data_matrix(dataset, apply_transformations = FALSE)
#'
#' # Get specific run with verbose transformation output
#' run1_data <- get_data_matrix(dataset, run_id = 1, verbose = TRUE)
#' }
#'
#' @export
get_data_matrix <- function(dataset, run_id = NULL, apply_transformations = TRUE,
                           apply_preprocessing = NULL,
                           verbose = FALSE, ...) {
  if (!is.fmri_dataset(dataset)) {
    stop("Object is not an fmri_dataset")
  }

  # ---------------------------------------------------------------------------
  # Backwards compatibility with legacy 'apply_preprocessing' argument
  # ---------------------------------------------------------------------------
  if (!is.null(apply_preprocessing)) {
    apply_transformations <- apply_preprocessing
  }
  
  # ============================================================================
  # Load Raw Data
  # ============================================================================
  
  # Try to get from cache first
  cache_key <- paste0("data_matrix_", ifelse(is.null(run_id), "all", paste(run_id, collapse = "_")))
  if (exists(cache_key, envir = dataset$data_cache)) {
    raw_data <- get(cache_key, envir = dataset$data_cache)
  } else {
    raw_data <- load_raw_data_matrix(dataset, run_id)
    assign(cache_key, raw_data, envir = dataset$data_cache)
  }
  
  # ============================================================================
  # Apply Spatial Masking
  # ============================================================================
  
  masked_data <- apply_spatial_mask(raw_data, dataset)
  
  # ============================================================================
  # Apply Transformation Pipeline
  # ============================================================================
  
  if (apply_transformations) {
    pipeline <- get_transformation_pipeline(dataset)
    if (!is.null(pipeline)) {
      if (verbose) {
        cat("Applying transformation pipeline...\n")
      }
      masked_data <- apply_pipeline(pipeline, masked_data, verbose = verbose, ...)
    }
  }
  
  # ============================================================================
  # Apply Temporal Censoring
  # ============================================================================
  
  final_data <- apply_temporal_censoring(masked_data, dataset, run_id)
  
  final_data
}

#' Get Mask Volume from fmri_dataset
#'
#' **Ticket #11**: Loads and returns the spatial mask with lazy loading support.
#'
#' @param x An `fmri_dataset` object
#' @param as_vector Logical indicating whether to return as vector (TRUE) or volume object (FALSE, default)
#' @param force_reload Logical indicating whether to bypass cache (default: FALSE)
#' @return NeuroVol mask object, logical vector, or NULL if no mask
#' 
#' @details
#' This function handles mask loading for different dataset types:
#' \itemize{
#'   \item \strong{file_vec/bids_file}: Loads mask from file using neuroim2
#'   \item \strong{memory_vec/bids_mem}: Returns pre-loaded mask object
#'   \item \strong{matrix}: Returns mask vector directly
#' }
#' 
#' @export
#' @family fmri_dataset
get_mask_volume <- function(x, as_vector = FALSE, force_reload = FALSE) {
  
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  # Check if mask exists
  if (is.null(x$mask_path) && is.null(x$mask_object) && is.null(x$mask_vector)) {
    return(NULL)
  }
  
  cache_key <- paste0("mask_", ifelse(as_vector, "vector", "volume"))
  
  # Check cache first
  if (!force_reload && exists(cache_key, envir = x$data_cache)) {
    return(get(cache_key, envir = x$data_cache))
  }
  
  # Load mask based on type
  if (!is.null(x$mask_path)) {
    # Load from file
    if (!check_package_available("neuroim2", "loading mask files", error = FALSE)) {
      stop("neuroim2 package is required to load mask files")
    }
    
    mask_vol <- neuroim2::read_vol(x$mask_path)
    
    if (as_vector) {
      mask_result <- as.vector(mask_vol)
    } else {
      mask_result <- mask_vol
    }
    
  } else if (!is.null(x$mask_object)) {
    # Pre-loaded object
    if (as_vector) {
      mask_result <- as.vector(x$mask_object)
    } else {
      mask_result <- x$mask_object
    }
    
  } else if (!is.null(x$mask_vector)) {
    # Vector mask (for matrix datasets)
    if (as_vector) {
      mask_result <- x$mask_vector
    } else {
      warning("Cannot convert vector mask to volume object - returning vector")
      mask_result <- x$mask_vector
    }
  }
  
  # Cache the result
  assign(cache_key, mask_result, envir = x$data_cache)
  
  return(mask_result)
}

#' Get Sampling Frame from fmri_dataset
#'
#' **Ticket #12**: Returns the first-class sampling_frame object.
#'
#' @param x An `fmri_dataset` object
#' @return A `sampling_frame` object
#' @export
#' @family fmri_dataset
get_sampling_frame <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  if (is.null(x$sampling_frame)) {
    stop("No sampling_frame found in fmri_dataset")
  }
  
  return(x$sampling_frame)
}

#' Get Event Table from fmri_dataset
#'
#' **Ticket #12**: Returns the event table.
#'
#' @param x An `fmri_dataset` object
#' @return A data.frame/tibble of events, or NULL if no events
#' @export
#' @family fmri_dataset
get_event_table <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  return(x$event_table)
}

#' Get TR from fmri_dataset
#'
#' **Ticket #12**: Convenience function to get TR from sampling_frame.
#'
#' @param x An `fmri_dataset` object
#' @param ... Additional arguments (ignored)
#' @return Numeric TR value
#' @export
#' @family fmri_dataset
get_TR.fmri_dataset <- function(x, ...) {
  validate_fmri_dataset_structure(x)
  return(x$sampling_frame$TR[1])
}

#' Get Run Lengths from fmri_dataset
#'
#' **Ticket #12**: Convenience function to get run lengths from sampling_frame.
#'
#' @param x An `fmri_dataset` object
#' @param ... Additional arguments (ignored)
#' @return Integer vector of run lengths
#' @export
#' @family fmri_dataset
get_run_lengths.fmri_dataset <- function(x, ...) {
  validate_fmri_dataset_structure(x)
  return(x$sampling_frame$run_lengths)
}

#' Get Number of Runs from fmri_dataset
#'
#' **Ticket #12**: Convenience function to get number of runs from sampling_frame.
#'
#' @param x An `fmri_dataset` object
#' @param ... Additional arguments (ignored)
#' @return Integer number of runs
#' @export
#' @family fmri_dataset
n_runs.fmri_dataset <- function(x, ...) {
  validate_fmri_dataset_structure(x)
  return(length(x$sampling_frame$run_lengths))
}

#' Get Number of Runs
#'
#' Convenience wrapper returning the number of runs in a dataset.
#'
#' @param x An `fmri_dataset` object
#' @return Integer number of runs
#' @export
get_num_runs <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }

  n_runs(get_sampling_frame(x))
}

#' Get Number of Voxels from fmri_dataset
#'
#' **Ticket #13**: Returns number of voxels after masking.
#'
#' @param x An `fmri_dataset` object
#' @return Integer number of voxels
#' @export
#' @family fmri_dataset
get_num_voxels <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  # If we have a mask, count TRUE voxels
  mask <- get_mask_volume(x, as_vector = TRUE)
  if (!is.null(mask)) {
    return(sum(mask))
  }
  
  # Otherwise, get from data dimensions
  dataset_type <- x$metadata$dataset_type
  
  if (dataset_type == "matrix") {
    return(ncol(x$image_matrix))
    
  } else if (dataset_type == "memory_vec") {
    # Get from first image object
    if (check_package_available("neuroim2", error = FALSE)) {
      dims <- dim(x$image_objects[[1]])
      return(prod(dims[-length(dims)]))  # All but time dimension
    } else {
      warning("Cannot determine voxel count without neuroim2")
      return(NA_integer_)
    }
    
  } else if (dataset_type %in% c("file_vec", "bids_file")) {
    # Would need to read header - expensive, so load a small sample
    if (check_package_available("neuroim2", error = FALSE)) {
      vol_info <- neuroim2::read_vol(x$image_paths[1])
      dims <- dim(vol_info)
      return(prod(dims[-length(dims)]))  # All but time dimension
    } else {
      warning("Cannot determine voxel count without neuroim2")
      return(NA_integer_)
    }
    
  } else {
    warning("Cannot determine voxel count for dataset_type: ", dataset_type)
    return(NA_integer_)
  }
}

#' Get Number of Timepoints from fmri_dataset
#'
#' **Ticket #13**: Returns total or per-run timepoints from sampling_frame.
#'
#' @param x An `fmri_dataset` object
#' @param ... Additional arguments (ignored)
#' @return Integer total number of timepoints
#' @export
#' @family fmri_dataset
n_timepoints.fmri_dataset <- function(x, ...) {
  validate_fmri_dataset_structure(x)
  return(sum(x$sampling_frame$run_lengths))
}

#' Get Number of Timepoints
#'
#' Convenience wrapper returning total or per-run timepoints.
#'
#' @param x An `fmri_dataset` object
#' @param run_id Optional integer vector of run IDs
#' @return Integer number of timepoints
#' @export
get_num_timepoints <- function(x, run_id = NULL) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }

  n_timepoints(get_sampling_frame(x), run_id)
}

#' Get Censor Vector from fmri_dataset
#'
#' **Ticket #14**: Returns the temporal censoring vector.
#'
#' @param x An `fmri_dataset` object
#' @return Logical or numeric vector, or NULL if no censoring
#' @export
#' @family fmri_dataset
get_censor_vector <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  return(x$censor_vector)
}

#' Get Metadata from fmri_dataset
#'
#' **Ticket #14**: Accesses metadata list with optional field selection.
#'
#' @param x An `fmri_dataset` object
#' @param field Character string specifying metadata field, or NULL for all metadata
#' @return Metadata list or specific field value
#' @export
#' @family fmri_dataset
get_metadata <- function(x, field = NULL) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  if (is.null(field)) {
    return(x$metadata)
  } else {
    if (field %in% names(x$metadata)) {
      return(x$metadata[[field]])
    } else {
      stop("Metadata field '", field, "' not found. Available fields: ", 
           paste(names(x$metadata), collapse = ", "))
    }
  }
}

#' Get Dataset Type from fmri_dataset
#'
#' **Ticket #14**: Returns the dataset type.
#'
#' @param x An `fmri_dataset` object
#' @return Character string indicating dataset type
#' @export
#' @family fmri_dataset
get_dataset_type <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  return(x$metadata$dataset_type)
}

#' Get Image Source Type from fmri_dataset
#'
#' Returns the class/type of the primary image data source.
#'
#' @param x An `fmri_dataset` object
#' @return Character string indicating source type ("character", "list", "matrix")
#' @export
#' @family fmri_dataset
get_image_source_type <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  # Determine source type based on what's populated
  if (!is.null(x$image_paths)) {
    return("character")
  } else if (!is.null(x$image_objects)) {
    return("list")
  } else if (!is.null(x$image_matrix)) {
    return("matrix")
  } else {
    stop("No image source found")
  }
}

# ============================================================================
# Internal Helper Functions for Data Loading and Processing
# ============================================================================

#' Load Raw Data Matrix
#'
#' Internal helper to load raw data based on dataset type.
#'
#' @param x An fmri_dataset object
#' @return Raw data matrix (time x voxels)
#' @keywords internal
#' @noRd
load_raw_data_matrix <- function(x, run_id = NULL) {

  cache_key <- paste0("raw_data_matrix_", ifelse(is.null(run_id), "all", paste(run_id, collapse = "_")))

  # Check cache first
  if (exists(cache_key, envir = x$data_cache)) {
    return(get(cache_key, envir = x$data_cache))
  }
  
  dataset_type <- x$metadata$dataset_type
  
  if (dataset_type == "matrix") {
    # Matrix data - already in memory
    raw_data <- x$image_matrix
    if (!is.null(run_id)) {
      raw_data <- extract_run_data(raw_data, x, run_id)
    }

  } else if (dataset_type == "memory_vec") {
    # Pre-loaded NeuroVec objects
    objs <- x$image_objects
    if (!is.null(run_id)) {
      objs <- objs[run_id]
    }
    raw_data <- extract_data_from_neurovecs(objs)

  } else if (dataset_type %in% c("file_vec", "bids_file")) {
    # Load from files
    paths <- x$image_paths
    if (!is.null(run_id)) {
      paths <- paths[run_id]
    }
    raw_data <- load_data_from_files(paths)

  } else if (dataset_type == "bids_mem") {
    # BIDS with preloaded data (should be in image_objects)
    if (!is.null(x$image_objects)) {
      objs <- x$image_objects
      if (!is.null(run_id)) {
        objs <- objs[run_id]
      }
      raw_data <- extract_data_from_neurovecs(objs)
    } else {
      # Fall back to loading from paths
      paths <- x$image_paths
      if (!is.null(run_id)) {
        paths <- paths[run_id]
      }
      raw_data <- load_data_from_files(paths)
    }
    
  } else {
    stop("Unknown dataset_type: ", dataset_type)
  }
  
  # Cache the raw data
  assign(cache_key, raw_data, envir = x$data_cache)
  
  return(raw_data)
}

#' Extract Data from NeuroVec Objects
#'
#' @param neurovecs List of NeuroVec/NeuroVol objects
#' @return Data matrix (time x voxels)
#' @keywords internal
#' @noRd
extract_data_from_neurovecs <- function(neurovecs) {
  
  if (!check_package_available("neuroim2", "extracting data from NeuroVec objects", error = TRUE)) {
    stop("neuroim2 is required to extract data from NeuroVec objects")
  }
  
  # Extract data from each run and concatenate
  run_matrices <- lapply(neurovecs, function(vol) {
    # Convert to matrix: spatial dimensions flattened, time as rows
    dims <- dim(vol)
    n_timepoints <- dims[length(dims)]
    n_voxels <- prod(dims[-length(dims)])
    
    # Reshape and transpose to get time x voxels
    vol_array <- as.array(vol)
    vol_matrix <- array(vol_array, dim = c(n_voxels, n_timepoints))
    return(t(vol_matrix))  # transpose to time x voxels
  })
  
  # Concatenate runs vertically (rbind)
  combined_matrix <- do.call(rbind, run_matrices)
  
  return(combined_matrix)
}

#' Load Data from Files
#'
#' @param file_paths Character vector of file paths
#' @return Data matrix (time x voxels)
#' @keywords internal
#' @noRd
load_data_from_files <- function(file_paths) {
  
  if (!check_package_available("neuroim2", "loading data from files", error = TRUE)) {
    stop("neuroim2 is required to load data from files")
  }
  
  # Load each file and extract data
  run_matrices <- lapply(file_paths, function(filepath) {
    vol <- neuroim2::read_vol(filepath)
    
    # Convert to matrix: spatial dimensions flattened, time as rows
    dims <- dim(vol)
    n_timepoints <- dims[length(dims)]
    n_voxels <- prod(dims[-length(dims)])
    
    # Reshape and transpose to get time x voxels
    vol_array <- as.array(vol)
    vol_matrix <- array(vol_array, dim = c(n_voxels, n_timepoints))
    return(t(vol_matrix))  # transpose to time x voxels
  })
  
  # Concatenate runs vertically (rbind)
  combined_matrix <- do.call(rbind, run_matrices)
  
  return(combined_matrix)
}

#' Apply Spatial Mask
#'
#' @param data_matrix Data matrix (time x voxels)
#' @param x fmri_dataset object
#' @return Masked data matrix
#' @keywords internal
#' @noRd
apply_spatial_mask <- function(data_matrix, x) {
  
  mask <- get_mask_volume(x, as_vector = TRUE)
  
  if (is.null(mask)) {
    return(data_matrix)
  }
  
  # Apply mask to columns (voxels)
  if (length(mask) != ncol(data_matrix)) {
    stop("Mask length (", length(mask), ") does not match number of voxels (", ncol(data_matrix), ")")
  }
  
  # Subset columns where mask is TRUE
  masked_data <- data_matrix[, mask, drop = FALSE]
  
  return(masked_data)
}

#' Apply Temporal Censoring
#'
#' @param data_matrix Data matrix (time x voxels)
#' @param x fmri_dataset object
#' @return Censored data matrix
#' @keywords internal
#' @noRd
apply_temporal_censoring <- function(data_matrix, x, run_id = NULL) {

  censor_vector <- get_censor_vector(x)

  if (is.null(censor_vector)) {
    return(data_matrix)
  }

  # Subset censor vector if run_id is provided
  if (!is.null(run_id)) {
    run_lengths <- get_run_lengths(x)
    cum_lengths <- cumsum(c(0, run_lengths))
    idx <- unlist(lapply(run_id, function(rid) seq(cum_lengths[rid] + 1, cum_lengths[rid + 1])))
    censor_vector <- censor_vector[idx]
  }

  # Convert to logical if numeric
  censor_logical <- if (is.numeric(censor_vector)) as.logical(censor_vector) else censor_vector

  # Check length
  if (length(censor_logical) != nrow(data_matrix)) {
    stop("Censor vector length (", length(censor_logical),
         ") does not match number of timepoints (", nrow(data_matrix), ")")
  }

  # Subset rows where censor is TRUE (keep these timepoints)
  censored_data <- data_matrix[censor_logical, , drop = FALSE]

  return(censored_data)
}

#' Apply Data Preprocessing
#'
#' @param data_matrix Data matrix (time x voxels)
#' @param x fmri_dataset object
#' @return Preprocessed data matrix
#' @keywords internal
#' @noRd
apply_data_preprocessing <- function(data_matrix, x) {
  
  processed_data <- data_matrix
  
  # Apply temporal z-scoring
  if (x$metadata$matrix_options$temporal_zscore) {
    processed_data <- apply(processed_data, 2, function(ts) {
      (ts - mean(ts, na.rm = TRUE)) / sd(ts, na.rm = TRUE)
    })
  }
  
  # Apply voxelwise detrending
  if (x$metadata$matrix_options$voxelwise_detrend) {
    time_vec <- seq_len(nrow(processed_data))
    processed_data <- apply(processed_data, 2, function(ts) {
      if (any(is.na(ts))) {
        return(ts)  # Skip if NAs present
      }
      lm_fit <- lm(ts ~ time_vec)
      return(residuals(lm_fit))
    })
  }
  
  return(processed_data)
}


#' Extract Run Data
#'
#' @param data_matrix Full data matrix (time x voxels)
#' @param x fmri_dataset object
#' @param run_id Integer vector of run IDs to extract
#' @return Data matrix with only specified runs
#' @keywords internal
#' @noRd
extract_run_data <- function(data_matrix, x, run_id) {
  
  if (is.null(run_id)) {
    return(data_matrix)
  }
  
  run_lengths <- get_run_lengths(x)
  n_runs <- length(run_lengths)
  
  # Validate run_id
  if (any(run_id < 1) || any(run_id > n_runs)) {
    stop("run_id must be between 1 and ", n_runs)
  }
  
  # Calculate cumulative timepoint indices for each run
  cum_lengths <- cumsum(c(0, run_lengths))
  
  # Extract timepoints for specified runs
  all_indices <- c()
  for (rid in run_id) {
    start_idx <- cum_lengths[rid] + 1
    end_idx <- cum_lengths[rid + 1]
    run_indices <- start_idx:end_idx
    all_indices <- c(all_indices, run_indices)
  }
  
  # Extract the specified timepoints
  extracted_data <- data_matrix[all_indices, , drop = FALSE]
  
  return(extracted_data)
} 