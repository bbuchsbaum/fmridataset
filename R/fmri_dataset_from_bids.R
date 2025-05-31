#' Create fmri_dataset from BIDS Projects
#'
#' This file implements the `as.fmri_dataset.bids_project()` method for creating
#' `fmri_dataset` objects from BIDS (Brain Imaging Data Structure) projects.
#' This method leverages the `bidser` package for BIDS data handling.
#'
#' @name fmri_dataset_from_bids
NULL

#' Convert BIDS Project to fmri_dataset
#'
#' Creates an `fmri_dataset` object from a BIDS project by automatically
#' extracting relevant files, metadata, and run information. This method
#' provides the most automated approach to creating datasets from standardized
#' neuroimaging data.
#'
#' @param x A `bids_project` object from the `bidser` package
#' @param subject_id Character string specifying the subject ID (required)
#' @param task_id Character string specifying the task ID, or NULL to auto-detect.
#'   Must be a single, non-NA string if provided.
#' @param session_id Character string specifying the session ID, or NULL for no session.
#'   Must be a single, non-NA string if provided.
#' @param run_ids Numeric vector of run IDs to include, or NULL for all runs.
#'   Must contain positive integers when supplied.
#' @param image_type Character string indicating which image type to use:
#'   "auto" (default), "raw", "preproc", or specific preprocessing pipeline name.
#'   Must be a single string; unrecognised values are treated as pipeline names.
#' @param event_table_source Character string indicating event source:
#'   "auto" (default), "events" (BIDS events.tsv), "none", or path to custom TSV.
#'   Must be a single string.
#' @param preload_data Logical indicating whether to preload image data (default: FALSE)
#' @param temporal_zscore Logical indicating whether to apply temporal z-scoring (default: FALSE)
#' @param voxelwise_detrend Logical indicating whether to apply voxelwise detrending (default: FALSE)
#' @param metadata List of additional metadata to include
#' @param ... Additional arguments (currently unused)
#' 
#' @return An `fmri_dataset` object with dataset_type "bids_file" or "bids_mem"
#' 
#' @details
#' This method performs the following operations:
#' \itemize{
#'   \item Validates that `bidser` package is available
#'   \item Extracts functional scan files based on subject, task, session, and run criteria
#'   \item Determines run lengths by reading NIfTI headers (requires `neuroim2`)
#'   \item Finds appropriate brain mask from BIDS derivatives
#'   \item Loads event tables from BIDS events.tsv files
#'   \item Populates comprehensive BIDS metadata
#' }
#'
#' The dataset type ("bids_file" or "bids_mem") is determined using
#' \code{determine_dataset_type(..., is_bids = TRUE, preload = preload_data)} and
#' stored in \code{dataset$metadata$dataset_type} of the returned object.
#'
#' **Image Type Selection (Subtask #9.2):**
#' - "auto": Prefers preprocessed images if available, falls back to raw
#' - "raw": Uses raw functional images from main BIDS directory
#' - "preproc": Uses preprocessed images from derivatives
#' - Pipeline name: Uses images from specific preprocessing pipeline
#' 
#' **Run Length Detection (Subtask #9.1):**
#' Automatically determines the number of timepoints in each run by reading
#' NIfTI headers using `neuroim2`. This eliminates the need to manually specify
#' `run_lengths` for BIDS datasets.
#' 
#' @examples
#' \dontrun{
#' library(bidser)
#' 
#' # Load BIDS project
#' bids_proj <- bids_project("/path/to/bids/dataset")
#' 
#' # Create dataset with auto-detection
#' dataset <- as.fmri_dataset(
#'   bids_proj,
#'   subject_id = "01",
#'   task_id = "rest"
#' )
#' 
#' # Create dataset with specific parameters
#' dataset <- as.fmri_dataset(
#'   bids_proj,
#'   subject_id = "01",
#'   task_id = "task",
#'   session_id = "ses-01",
#'   run_ids = c(1, 2),
#'   image_type = "preproc",
#'   preload_data = TRUE
#' )
#' }
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_dataset_create}} for the primary constructor
as.fmri_dataset.bids_project <- function(x, subject_id,
                                        task_id = NULL,
                                        session_id = NULL,
                                        run_ids = NULL,
                                        image_type = "auto",
                                        event_table_source = "auto",
                                        preload_data = FALSE,
                                        temporal_zscore = FALSE,
                                        voxelwise_detrend = FALSE,
                                        metadata = list(),
                                        ...) {
  
  # Check that bidser is available
  check_package_available("bidser", "processing BIDS projects", error = TRUE)
  
  # Validate required arguments
  if (missing(subject_id)) {
    stop("subject_id is required for BIDS-based fmri_dataset")
  }
  
  if (!is.character(subject_id) || length(subject_id) != 1) {
    stop("subject_id must be a single character string")
  }

  # Validate task_id if provided
  if (!is.null(task_id)) {
    if (!is.character(task_id) || length(task_id) != 1 || is.na(task_id)) {
      stop("task_id must be a single, non-NA character string or NULL")
    }
  }

  # Validate session_id if provided
  if (!is.null(session_id)) {
    if (!is.character(session_id) || length(session_id) != 1 || is.na(session_id)) {
      stop("session_id must be a single, non-NA character string or NULL")
    }
  }

  # Validate run_ids if provided
  if (!is.null(run_ids)) {
    if (!is.numeric(run_ids) || any(is.na(run_ids)) || any(run_ids <= 0)) {
      stop("run_ids must be a numeric vector of positive integers or NULL")
    }
    run_ids <- as.integer(run_ids)
  }

  # Validate image_type
  if (!is.character(image_type) || length(image_type) != 1 || is.na(image_type)) {
    stop("image_type must be a single, non-NA character string")
  }

  valid_image_types <- c("auto", "raw", "preproc")
  if (!image_type %in% valid_image_types) {
    warning("Unrecognised image_type '", image_type, "' - treating as pipeline name")
  }

  # Validate event_table_source
  if (!is.character(event_table_source) || length(event_table_source) != 1 ||
      is.na(event_table_source)) {
    stop("event_table_source must be a single, non-NA character string")
  }
  
  # Extract functional scans (Subtask #9.2)
  func_scans <- extract_functional_scans(x, subject_id, task_id, session_id, run_ids, image_type)
  
  # Get TR from BIDS metadata
  TR <- extract_bids_TR(x, func_scans)
  
  # Determine run lengths from NIfTI headers (Subtask #9.1)
  run_lengths <- determine_bids_run_lengths(func_scans$file_paths)
  
  # Extract brain mask
  mask_info <- extract_bids_mask(x, subject_id, session_id, image_type)
  
  # Extract event table
  event_table <- extract_bids_events(x, subject_id, task_id, session_id, run_ids, event_table_source)
  
  # Prepare BIDS metadata
  bids_metadata <- prepare_bids_metadata(x, func_scans, subject_id, task_id, session_id, run_ids, image_type)
  
  # Merge with user metadata
  final_metadata <- c(metadata, list(bids_info = bids_metadata))
  
  # Call the primary constructor
  dataset <- fmri_dataset_create(
    images = func_scans$file_paths,
    mask = mask_info$file_path,
    TR = TR,
    run_lengths = run_lengths,
    event_table = event_table,
    censor_vector = NULL,  # Could be extracted from BIDS confounds in future
    base_path = dirname(func_scans$file_paths[1]),
    image_mode = "normal",
    preload_data = preload_data,
    temporal_zscore = temporal_zscore,
    voxelwise_detrend = voxelwise_detrend,
    metadata = final_metadata,
    ...
  )

  # Determine dataset type for BIDS source and store
  dataset$metadata$dataset_type <- determine_dataset_type(
    func_scans$file_paths,
    mask_info$file_path,
    is_bids = TRUE,
    preload = preload_data
  )

  dataset
}

#' Extract Functional Scans from BIDS Project
#'
#' **Subtask #9.2**: Logic for selecting raw vs. preprocessed images based on image_type
#'
#' @param bids_proj BIDS project object
#' @param subject_id Subject ID
#' @param task_id Task ID (can be NULL for auto-detection)
#' @param session_id Session ID (can be NULL)
#' @param run_ids Run IDs (can be NULL for all)
#' @param image_type Type of images to extract
#' @return List with file_paths and metadata
#' @keywords internal
#' @noRd
extract_functional_scans <- function(bids_proj, subject_id, task_id, session_id, run_ids, image_type) {
  
  tryCatch({
    
    if (image_type == "auto") {
      # Try preprocessed first, fall back to raw
      preproc_scans <- try_extract_preprocessed_scans(bids_proj, subject_id, task_id, session_id, run_ids)
      if (!is.null(preproc_scans)) {
        return(list(
          file_paths = preproc_scans,
          source_type = "preproc_auto"
        ))
      } else {
        raw_scans <- extract_raw_scans(bids_proj, subject_id, task_id, session_id, run_ids)
        return(list(
          file_paths = raw_scans,
          source_type = "raw_fallback"
        ))
      }
      
    } else if (image_type == "raw") {
      # Extract raw functional scans
      raw_scans <- extract_raw_scans(bids_proj, subject_id, task_id, session_id, run_ids)
      return(list(
        file_paths = raw_scans,
        source_type = "raw_explicit"
      ))
      
    } else if (image_type == "preproc") {
      # Extract preprocessed scans (generic)
      preproc_scans <- try_extract_preprocessed_scans(bids_proj, subject_id, task_id, session_id, run_ids)
      if (is.null(preproc_scans)) {
        stop("No preprocessed images found for subject ", subject_id)
      }
      return(list(
        file_paths = preproc_scans,
        source_type = "preproc_explicit"
      ))
      
    } else {
      # Specific pipeline name
      pipeline_scans <- extract_pipeline_scans(bids_proj, subject_id, task_id, session_id, run_ids, image_type)
      return(list(
        file_paths = pipeline_scans,
        source_type = paste0("pipeline_", image_type)
      ))
    }
    
  }, error = function(e) {
    stop("Failed to extract functional scans: ", e$message)
  })
}

#' Extract Raw Functional Scans
#'
#' @param bids_proj BIDS project object  
#' @param subject_id Subject ID
#' @param task_id Task ID
#' @param session_id Session ID
#' @param run_ids Run IDs
#' @return Character vector of file paths
#' @keywords internal
#' @noRd
extract_raw_scans <- function(bids_proj, subject_id, task_id, session_id, run_ids) {
  
  # Use bidser::func_scans to get raw functional scans
  if (requireNamespace("bidser", quietly = TRUE)) {
    scans <- bidser::func_scans(bids_proj, 
                               subject_id = subject_id,
                               task_id = task_id,
                               session_id = session_id,
                               run_ids = run_ids)
    
    if (length(scans) == 0) {
      stop("No raw functional scans found for subject ", subject_id)
    }
    
    return(scans)
  } else {
    stop("bidser package is required but not available")
  }
}

#' Try to Extract Preprocessed Scans
#'
#' @param bids_proj BIDS project object
#' @param subject_id Subject ID
#' @param task_id Task ID
#' @param session_id Session ID  
#' @param run_ids Run IDs
#' @return Character vector of file paths or NULL if not found
#' @keywords internal
#' @noRd
try_extract_preprocessed_scans <- function(bids_proj, subject_id, task_id, session_id, run_ids) {
  
  if (requireNamespace("bidser", quietly = TRUE)) {
    tryCatch({
      # Try to get preprocessed scans
      scans <- bidser::preproc_scans(bids_proj,
                                    subject_id = subject_id,
                                    task_id = task_id,
                                    session_id = session_id,
                                    run_ids = run_ids)
      
      if (length(scans) > 0) {
        return(scans)
      } else {
        return(NULL)
      }
    }, error = function(e) {
      # If preproc_scans fails, return NULL to fall back
      return(NULL)
    })
  } else {
    return(NULL)
  }
}

#' Extract Pipeline-Specific Scans
#'
#' @param bids_proj BIDS project object
#' @param subject_id Subject ID
#' @param task_id Task ID
#' @param session_id Session ID
#' @param run_ids Run IDs
#' @param pipeline_name Pipeline name
#' @return Character vector of file paths
#' @keywords internal
#' @noRd
extract_pipeline_scans <- function(bids_proj, subject_id, task_id, session_id, run_ids, pipeline_name) {
  
  if (requireNamespace("bidser", quietly = TRUE)) {
    tryCatch({
      # Try to get scans from specific pipeline
      scans <- bidser::preproc_scans(bids_proj,
                                    subject_id = subject_id,
                                    task_id = task_id,
                                    session_id = session_id,
                                    run_ids = run_ids,
                                    pipeline = pipeline_name)
      
      if (length(scans) == 0) {
        stop("No scans found for pipeline '", pipeline_name, "' and subject ", subject_id)
      }
      
      return(scans)
    }, error = function(e) {
      stop("Failed to extract scans from pipeline '", pipeline_name, "': ", e$message)
    })
  } else {
    stop("bidser package is required but not available")
  }
}

#' Determine Run Lengths from BIDS Files
#'
#' **Subtask #9.1**: Logic for determining run_lengths from NIfTI headers via neuroim2
#'
#' @param file_paths Character vector of NIfTI file paths
#' @return Integer vector of run lengths
#' @keywords internal
#' @noRd
determine_bids_run_lengths <- function(file_paths) {
  
  # Check if neuroim2 is available for reading headers
  if (!check_package_available("neuroim2", "reading NIfTI headers for run length detection", error = FALSE)) {
    stop("neuroim2 package is required to automatically determine run lengths from BIDS files.\n",
         "Install with: install.packages('neuroim2')\n",
         "Alternatively, use fmri_dataset_create() and specify run_lengths manually.")
  }
  
  run_lengths <- integer(length(file_paths))
  
  for (i in seq_along(file_paths)) {
    tryCatch({
      # Read NIfTI header to get dimensions
      vol_info <- neuroim2::read_vol(file_paths[i])
      dims <- dim(vol_info)
      
      # Fourth dimension should be time
      if (length(dims) >= 4) {
        run_lengths[i] <- dims[4]
      } else {
        stop("Image file does not have a time dimension: ", file_paths[i])
      }
      
    }, error = function(e) {
      stop("Failed to read image header for ", basename(file_paths[i]), ": ", e$message)
    })
  }
  
  if (any(run_lengths <= 0)) {
    invalid_files <- file_paths[run_lengths <= 0]
    stop("Invalid run lengths detected for files: ", paste(basename(invalid_files), collapse = ", "))
  }
  
  return(run_lengths)
}

#' Extract TR from BIDS Metadata
#'
#' @param bids_proj BIDS project object
#' @param func_scans Functional scan information
#' @return Numeric TR value
#' @keywords internal
#' @noRd
extract_bids_TR <- function(bids_proj, func_scans) {
  
  if (requireNamespace("bidser", quietly = TRUE)) {
    tryCatch({
      # Use bidser to get repetition time
      TR <- bidser::get_repetition_time(bids_proj, func_scans$file_paths[1])
      
      if (is.null(TR) || is.na(TR) || TR <= 0) {
        stop("Invalid or missing TR in BIDS metadata")
      }
      
      return(TR)
      
    }, error = function(e) {
      stop("Failed to extract TR from BIDS metadata: ", e$message)
    })
  } else {
    stop("bidser package is required but not available")
  }
}

#' Extract Brain Mask from BIDS
#'
#' @param bids_proj BIDS project object
#' @param subject_id Subject ID
#' @param session_id Session ID
#' @param image_type Image type for mask compatibility
#' @return List with file_path and metadata
#' @keywords internal
#' @noRd
extract_bids_mask <- function(bids_proj, subject_id, session_id, image_type) {
  
  if (requireNamespace("bidser", quietly = TRUE)) {
    tryCatch({
      # Try to get brain mask
      mask_path <- bidser::brain_mask(bids_proj, 
                                     subject_id = subject_id, 
                                     session_id = session_id)
      
      if (is.null(mask_path) || !file.exists(mask_path)) {
        warning("No brain mask found in BIDS derivatives for subject ", subject_id)
        return(list(file_path = NULL, source = "none"))
      }
      
      return(list(
        file_path = mask_path,
        source = "bids_derivatives"
      ))
      
    }, error = function(e) {
      warning("Failed to extract brain mask from BIDS: ", e$message)
      return(list(file_path = NULL, source = "error"))
    })
  } else {
    return(list(file_path = NULL, source = "no_bidser"))
  }
}

#' Extract Event Table from BIDS
#'
#' @param bids_proj BIDS project object
#' @param subject_id Subject ID
#' @param task_id Task ID
#' @param session_id Session ID
#' @param run_ids Run IDs
#' @param event_table_source Source specification
#' @return Data.frame/tibble or NULL
#' @keywords internal
#' @noRd
extract_bids_events <- function(bids_proj, subject_id, task_id, session_id, run_ids, event_table_source) {
  
  if (event_table_source == "none") {
    return(NULL)
  }
  
  if (event_table_source != "auto" && event_table_source != "events") {
    # Custom TSV file path
    if (file.exists(event_table_source)) {
      return(event_table_source)  # Return path, will be processed by fmri_dataset_create
    } else {
      warning("Custom event table file not found: ", event_table_source)
      return(NULL)
    }
  }
  
  # Extract BIDS events.tsv files
  if (requireNamespace("bidser", quietly = TRUE)) {
    tryCatch({
      events <- bidser::read_events(bids_proj,
                                   subject_id = subject_id,
                                   task_id = task_id,
                                   session_id = session_id,
                                   run_ids = run_ids)
      
      if (is.null(events) || nrow(events) == 0) {
        warning("No events found in BIDS for subject ", subject_id, ", task ", task_id)
        return(NULL)
      }
      
      return(events)
      
    }, error = function(e) {
      warning("Failed to extract events from BIDS: ", e$message)
      return(NULL)
    })
  } else {
    warning("bidser package not available - cannot extract BIDS events")
    return(NULL)
  }
}

#' Prepare BIDS Metadata
#'
#' @param bids_proj BIDS project object
#' @param func_scans Functional scan information
#' @param subject_id Subject ID
#' @param task_id Task ID
#' @param session_id Session ID
#' @param run_ids Run IDs
#' @param image_type Image type
#' @return List of BIDS metadata
#' @keywords internal
#' @noRd
prepare_bids_metadata <- function(bids_proj, func_scans, subject_id, task_id, session_id, run_ids, image_type) {
  
  # Get BIDS project path
  if (requireNamespace("bidser", quietly = TRUE)) {
    project_path <- tryCatch({
      bids_proj$path
    }, error = function(e) {
      "unknown"
    })
  } else {
    project_path <- "unknown"
  }
  
  # Infer run IDs from file paths if not specified
  if (is.null(run_ids)) {
    run_ids <- infer_run_ids_from_paths(func_scans$file_paths)
  }
  
  list(
    project_path = project_path,
    subject_id = subject_id,
    session_id = session_id,
    task_id = task_id,
    run_ids = run_ids,
    image_type_source = func_scans$source_type
  )
}

#' Infer Run IDs from File Paths
#'
#' @param file_paths Character vector of file paths
#' @return Integer vector of run IDs
#' @keywords internal
#' @noRd
infer_run_ids_from_paths <- function(file_paths) {
  
  # Extract run numbers from BIDS-compliant filenames
  run_matches <- regmatches(file_paths, regexpr("run-[0-9]+", file_paths))
  
  if (length(run_matches) > 0) {
    run_numbers <- as.integer(sub("run-", "", run_matches))
    return(run_numbers)
  } else {
    # If no run numbers found, assume sequential runs
    return(seq_along(file_paths))
  }
} 