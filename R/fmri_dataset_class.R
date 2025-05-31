#' fMRI Dataset S3 Class Definition
#'
#' This file defines the core S3 class structure for `fmri_dataset` objects.
#' The `fmri_dataset` is a unified container for fMRI data from various sources
#' including raw NIfTI files, BIDS projects, pre-loaded NeuroVec objects,
#' and in-memory matrices.
#'
#' @section Structure:
#' An `fmri_dataset` object is an S3 object (named list) with the following components:
#'
#' \describe{
#'   \item{image_paths}{Character vector: Full paths to NIfTI image files}
#'   \item{image_objects}{List: Pre-loaded NeuroVec/NeuroVol objects}
#'   \item{image_matrix}{Matrix: In-memory data matrix (time x voxels)}
#'   \item{mask_path}{Character: Full path to NIfTI mask file}
#'   \item{mask_object}{LogicalNeuroVol: Pre-loaded mask object}
#'   \item{mask_vector}{Logical vector/matrix for image_matrix}
#'   \item{sampling_frame}{sampling_frame object: TR, run_lengths, etc.}
#'   \item{event_table}{tibble: Event data (onset, duration, trial_type, etc.)}
#'   \item{censor_vector}{Logical or numeric vector for censoring}
#'   \item{metadata}{List: Descriptive and provenance metadata}
#'   \item{data_cache}{Environment: For memoized/loaded data}
#' }
#'
#' @section Dataset Types:
#' The `metadata$dataset_type` field indicates the primary data source:
#' \describe{
#'   \item{file_vec}{NIfTI files on disk}
#'   \item{memory_vec}{Pre-loaded NeuroVec objects}
#'   \item{matrix}{Plain in-memory matrix}
#'   \item{bids_file}{BIDS-derived with lazy loading}
#'   \item{bids_mem}{BIDS-derived with preloaded data}
#' }
#'
#' @name fmri_dataset-class
#' @family fmri_dataset
NULL

#' Create New fmri_dataset Object Structure
#'
#' Internal function to create the core structure of an fmri_dataset object.
#' This function creates the skeleton structure but does not validate inputs
#' or populate data - that is handled by the constructors.
#'
#' @return A bare `fmri_dataset` object with NULL/empty slots
#' @keywords internal
#' @noRd
new_fmri_dataset <- function() {
  structure(
    list(
      # --- Data Sources (Only one set will be populated) ---
      image_paths = NULL,      # Character vector: Full paths to NIfTI image files
      image_objects = NULL,    # List: Pre-loaded NeuroVec/NeuroVol objects
      image_matrix = NULL,     # Matrix: In-memory data matrix (time x voxels)

      mask_path = NULL,        # Character: Full path to NIfTI mask file
      mask_object = NULL,      # LogicalNeuroVol: Pre-loaded mask object
      mask_vector = NULL,      # Logical vector/matrix for image_matrix

      # --- Essential Metadata & Structure ---
      sampling_frame = NULL,   # 'sampling_frame' object (TR, run_lengths, etc.)
      event_table = NULL,      # 'tibble' of event data (onset, duration, trial_type, etc.)
      censor_vector = NULL,    # Optional: Logical or numeric vector for censoring

      # --- Descriptive & Provenance Metadata ---
      metadata = list(
        dataset_type = NULL,           # Character: "file_vec", "memory_vec", "matrix", "bids_file", "bids_mem"
        source_description = NULL,     # Character: Origin of the data
        TR = NULL,                     # Numeric: Repetition Time (convenience copy)
        base_path = NULL,              # Character: Base path for relative file paths
        
        # BIDS-specific metadata (populated by as.fmri_dataset.bids_project)
        bids_info = list(
          project_path = NULL,
          subject_id = NULL,
          session_id = NULL,
          task_id = NULL,
          run_ids = NULL,              # Could be multiple runs
          image_type_source = NULL     # e.g., "raw", "preproc"
        ),
        
        # File-loading options (if dataset_type %in% c("file_vec", "bids_file"))
        file_options = list(
          mode = "normal",             # Loading mode for files
          preload = FALSE              # Whether data is preloaded
        ),
        
        # Matrix preprocessing options
        matrix_options = list(
          temporal_zscore = FALSE,     # Apply temporal z-scoring
          voxelwise_detrend = FALSE    # Apply voxelwise detrending
        ),
        
        # User-provided extra metadata
        extra = list()
      ),

      # --- Internal Cache ---
      data_cache = new.env(hash = TRUE, parent = emptyenv()) # For memoized/loaded data
    ),
    class = "fmri_dataset"
  )
}

#' Check if Object is an fmri_dataset
#'
#' @param x Object to test
#' @return Logical indicating whether `x` is an `fmri_dataset`
#' @export
is.fmri_dataset <- function(x) {
  inherits(x, "fmri_dataset")
}

#' Internal Helper: Validate fmri_dataset Structure
#'
#' Validates that an fmri_dataset object has the correct structure and
#' that only one set of image sources is populated.
#'
#' @param x An fmri_dataset object
#' @return TRUE if valid, throws error if invalid
#' @keywords internal
#' @noRd
validate_fmri_dataset_structure <- function(x) {
  if (!is.fmri_dataset(x)) {
    stop("Object is not an fmri_dataset")
  }
  
  # Check that only one image source is populated
  image_sources <- c(
    !is.null(x$image_paths),
    !is.null(x$image_objects),
    !is.null(x$image_matrix)
  )
  
  if (sum(image_sources) != 1) {
    stop("Exactly one image source must be populated (paths, objects, or matrix)")
  }
  
  # Check that only one mask source is populated (if any)
  mask_sources <- c(
    !is.null(x$mask_path),
    !is.null(x$mask_object),
    !is.null(x$mask_vector)
  )
  
  if (sum(mask_sources) > 1) {
    stop("At most one mask source can be populated (path, object, or vector)")
  }
  
  # Check required metadata
  if (is.null(x$metadata$dataset_type)) {
    stop("dataset_type must be specified in metadata")
  }
  
  if (!x$metadata$dataset_type %in% VALID_DATASET_TYPES) {
    stop("Invalid dataset_type: ", x$metadata$dataset_type)
  }
  
  # Check that sampling_frame is present
  if (is.null(x$sampling_frame)) {
    stop("sampling_frame is required")
  }
  
  # Check that event_table is a data.frame/tibble if present
  if (!is.null(x$event_table) && !is.data.frame(x$event_table)) {
    stop("event_table must be a data.frame or tibble")
  }
  
  # Validate dataset type consistency using enhanced utils function
  validate_dataset_type_consistency(x$metadata$dataset_type, 
                                  get_primary_image_source(x), 
                                  get_primary_mask_source(x))
  
  TRUE
}

#' Internal Helper: Get Primary Image Source
#'
#' Returns the primary image data source from an fmri_dataset object
#'
#' @param x An fmri_dataset object
#' @return The primary image source (paths, objects, or matrix)
#' @keywords internal
#' @noRd
get_primary_image_source <- function(x) {
  if (!is.null(x$image_paths)) {
    return(x$image_paths)
  } else if (!is.null(x$image_objects)) {
    return(x$image_objects)
  } else if (!is.null(x$image_matrix)) {
    return(x$image_matrix)
  } else {
    stop("No image source found")
  }
}

#' Internal Helper: Get Primary Mask Source
#'
#' Returns the primary mask source from an fmri_dataset object
#'
#' @param x An fmri_dataset object
#' @return The primary mask source (path, object, vector, or NULL)
#' @keywords internal
#' @noRd
get_primary_mask_source <- function(x) {
  if (!is.null(x$mask_path)) {
    return(x$mask_path)
  } else if (!is.null(x$mask_object)) {
    return(x$mask_object)
  } else if (!is.null(x$mask_vector)) {
    return(x$mask_vector)
  } else {
    return(NULL)
  }
}

#' Internal Helper: Get Image Source Type
#'
#' Returns the class/type of the primary image data source
#'
#' @param x An fmri_dataset object
#' @return Character string indicating the source type
#' @keywords internal
#' @noRd
get_image_source_type <- function(x) {
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