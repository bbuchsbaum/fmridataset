#' Create an fMRI Dataset Object
#'
#' This is the primary constructor for creating `fmri_dataset` objects from various
#' data sources including file paths, pre-loaded objects, or matrices. It handles
#' input validation, type determination, and proper object initialization.
#'
#' @param images Primary data source. Can be:
#'   \itemize{
#'     \item Character vector: Paths to NIfTI files
#'     \item List: Pre-loaded NeuroVec/NeuroVol objects  
#'     \item Matrix/Array: In-memory data matrix (time x voxels)
#'   }
#' @param mask Mask specification. Can be:
#'   \itemize{
#'     \item Character: Path to NIfTI mask file
#'     \item NeuroVol: Pre-loaded mask object
#'     \item Logical vector/matrix: In-memory mask for matrix data
#'     \item NULL: No mask (all voxels included)
#'   }
#' @param TR Numeric scalar specifying repetition time in seconds.
#' @param run_lengths Numeric vector specifying the length of each run in timepoints.
#' @param event_table Optional. Event information as:
#'   \itemize{
#'     \item data.frame/tibble: Event data with onset, duration, trial_type columns
#'     \item Character: Path to TSV file containing events
#'     \item NULL: No events specified
#'   }
#' @param censor_vector Optional logical or numeric vector for censoring timepoints.
#' @param base_path Character. Base path for resolving relative file paths (default ".").
#' @param image_mode Character. Loading mode for file-based images: 
#'   "normal", "bigvec", "mmap", "filebacked" (default "normal").
#' @param preload_data Logical. Whether to immediately load file-based data into cache (default FALSE).
#' @param transformation_pipeline A transformation_pipeline object for data preprocessing.
#' @param temporal_zscore Logical. Whether to apply temporal z-scoring to matrix data (default FALSE).
#' @param voxelwise_detrend Logical. Whether to apply voxelwise detrending to matrix data (default FALSE).
#' @param metadata List. Additional user-provided metadata (default empty list).
#' @param ... Additional arguments for future expansion.
#'
#' @return An `fmri_dataset` object containing the structured fMRI data and metadata.
#'
#' @examples
#' \dontrun{
#' # From file paths
#' dset1 <- fmri_dataset_create(
#'   images = c("run1.nii", "run2.nii"),
#'   mask = "mask.nii",
#'   TR = 2.0,
#'   run_lengths = c(200, 200)
#' )
#' 
#' # From matrix data
#' mat <- matrix(rnorm(1000 * 100), 1000, 100)
#' dset2 <- fmri_dataset_create(
#'   images = mat,
#'   mask = NULL,
#'   TR = 2.5,
#'   run_lengths = 1000
#' )
#' 
#' # With events and preprocessing
#' events <- data.frame(onset = c(10, 50, 90), duration = c(2, 2, 2), 
#'                      trial_type = c("A", "B", "A"))
#' dset3 <- fmri_dataset_create(
#'   images = mat,
#'   mask = NULL,
#'   TR = 2.0,
#'   run_lengths = 1000,
#'   event_table = events,
#'   temporal_zscore = TRUE
#' )
#' }
#'
#' @export
fmri_dataset_create <- function(images, 
                               mask = NULL,
                               TR, 
                               run_lengths,
                               event_table = NULL,
                               censor_vector = NULL,
                               base_path = ".",
                               image_mode = c("normal", "bigvec", "mmap", "filebacked"),
                               preload_data = FALSE,
                               transformation_pipeline = NULL,
                               temporal_zscore = FALSE,
                               voxelwise_detrend = FALSE,
                               metadata = list(),
                               ...) {
  
  # Match and validate image_mode
  image_mode <- match.arg(image_mode)
  
  # === INPUT VALIDATION ===
  
  # Validate TR
  if (!is.numeric(TR) || length(TR) != 1 || TR <= 0) {
    stop("TR values must be positive")
  }
  
  # Validate run_lengths
  if (!is.numeric(run_lengths) || any(run_lengths <= 0)) {
    stop("run_lengths must be positive")
  }
  run_lengths <- as.integer(round(run_lengths))
  
  # Validate base_path
  if (!is.character(base_path) || length(base_path) != 1) {
    stop("base_path must be a character scalar")
  }
  
  # Validate logical flags
  if (!is.logical(preload_data) || length(preload_data) != 1) {
    stop("preload_data must be a logical scalar")
  }
  if (!is.logical(temporal_zscore) || length(temporal_zscore) != 1) {
    stop("temporal_zscore must be a logical scalar")
  }
  if (!is.logical(voxelwise_detrend) || length(voxelwise_detrend) != 1) {
    stop("voxelwise_detrend must be a logical scalar")
  }
  
  # Validate metadata
  if (!is.list(metadata)) {
    stop("metadata must be a list")
  }
  
  # Validate transformation pipeline
  if (!is.null(transformation_pipeline) && 
      !is.transformation_pipeline(transformation_pipeline)) {
    stop("transformation_pipeline must be a transformation_pipeline object or NULL")
  }
  
  # === DETERMINE DATASET TYPE ===
  dataset_type <- determine_dataset_type(images, mask, is_bids = FALSE, preload = preload_data)
  
  # === VALIDATE INPUTS BASED ON DATASET TYPE ===
  
  if (dataset_type == "file_vec") {
    # File-based validation
    if (!is.character(images)) {
      stop("For file_vec dataset_type, images must be character paths")
    }
    
    # Resolve to absolute paths
    images <- resolve_paths(images, base_path)
    
    # Validate files exist
    missing_files <- images[!file.exists(images)]
    if (length(missing_files) > 0) {
      stop("Image files not found: ", paste(missing_files, collapse = ", "))
    }
    
    # Handle mask path
    if (!is.null(mask)) {
      if (!is.character(mask) || length(mask) != 1) {
        stop("For file_vec dataset_type, mask must be a character path or NULL")
      }
      mask <- resolve_paths(mask, base_path)
      if (!file.exists(mask)) {
        stop("Mask file not found: ", mask)
      }
    }
    
  } else if (dataset_type == "memory_vec") {
    # Pre-loaded object validation
    if (!is.list(images)) {
      stop("For memory_vec dataset_type, images must be a list of NeuroVec objects")
    }

    # Validate that number of image objects matches run_lengths
    if (length(images) != length(run_lengths)) {
      stop("For memory_vec dataset_type, length(images) (", length(images),
           ") must match length(run_lengths) (", length(run_lengths), ")")
    }

    # TODO: Add NeuroVec class validation when neuroim2 is available
    # This would require conditional checking since neuroim2 is in Suggests
    
  } else if (dataset_type == "matrix") {
    # Matrix validation
    if (!is.matrix(images) && !is.array(images)) {
      stop("For matrix dataset_type, images must be a matrix or array")
    }
    
    # Ensure it's a matrix
    if (is.array(images) && length(dim(images)) != 2) {
      stop("For matrix dataset_type, images array must be 2-dimensional (time x voxels)")
    }
    images <- as.matrix(images)
    
    # Validate dimensions match run_lengths
    expected_timepoints <- sum(run_lengths)
    if (nrow(images) != expected_timepoints) {
      stop("Sum of run_lengths (", expected_timepoints, ") does not match matrix row count (", nrow(images), ")")
    }
    
    # Validate mask if provided
    if (!is.null(mask)) {
      if (!is.logical(mask) && !is.numeric(mask)) {
        stop("For matrix dataset_type, mask must be logical/numeric vector or NULL")
      }
      
      # Convert to logical if numeric and validate values
      if (is.numeric(mask)) {
        if (!all(mask %in% c(0, 1))) {
          stop("Numeric mask values must be 0 or 1")
        }
        mask <- as.logical(mask)
      }
      
      # Check dimensions
      if (length(mask) != ncol(images)) {
        stop("Mask length (", length(mask), ") does not match number of matrix columns (", 
             ncol(images), ")")
      }
    }
  }
  
  # === VALIDATE AND PROCESS EVENT TABLE ===
  if (!is.null(event_table)) {
    if (is.character(event_table)) {
      # Load from file
      event_table_path <- resolve_paths(event_table, base_path)
      if (!file.exists(event_table_path)) {
        stop("Event table file not found: ", event_table_path)
      }
      
      # Read TSV file - use read.delim for base R compatibility
      event_table <- read.delim(event_table_path, stringsAsFactors = FALSE)
    }
    
    # Ensure it's a data.frame
    if (!is.data.frame(event_table)) {
      stop("event_table must be a data.frame, tibble, or path to TSV file")
    }
    
    # Convert to tibble if available, otherwise keep as data.frame
    if (requireNamespace("tibble", quietly = TRUE)) {
      event_table <- tibble::as_tibble(event_table)
    }
    
    # Validate event table structure
    required_cols <- c("onset")
    missing_cols <- setdiff(required_cols, names(event_table))
    if (length(missing_cols) > 0) {
      stop("Event table missing required columns: ", paste(missing_cols, collapse = ", "))
    }
    
    # Validate onset times are within experiment bounds
    total_duration <- sum(run_lengths) * TR
    invalid_onsets <- event_table$onset[event_table$onset < 0 | event_table$onset >= total_duration]
    if (length(invalid_onsets) > 0) {
      warning("Some event onsets are outside experiment duration [0, ", total_duration, "): ",
              paste(head(invalid_onsets, 5), collapse = ", "))
    }
  }
  
  # === VALIDATE CENSOR VECTOR ===
  if (!is.null(censor_vector)) {
    if (!is.logical(censor_vector) && !is.numeric(censor_vector)) {
      stop("censor_vector must be logical, numeric, or NULL")
    }

    if (is.numeric(censor_vector) && !all(censor_vector %in% c(0, 1))) {
      stop("Numeric censor_vector values must be 0 or 1")
    }
    
    expected_length <- sum(run_lengths)
    if (length(censor_vector) != expected_length) {
      stop("censor_vector length (", length(censor_vector), 
           ") does not match total timepoints (", expected_length, ")")
    }
    
    # Keep the original type - don't convert to logical
    # Users might expect to get back the same type they passed in
  }
  
  # === CREATE SAMPLING FRAME ===
  sframe <- sampling_frame(run_lengths, TR)
  
  # === CREATE FMRI_DATASET OBJECT ===
  obj <- new_fmri_dataset()

  # Assign transformation pipeline if provided
  obj <- set_transformation_pipeline(obj, transformation_pipeline)
  
  # Populate data sources based on dataset_type
  if (dataset_type == "file_vec") {
    obj$image_paths <- images
    obj$mask_path <- mask
  } else if (dataset_type == "memory_vec") {
    obj$image_objects <- images
    obj$mask_object <- mask
  } else if (dataset_type == "matrix") {
    obj$image_matrix <- images
    obj$mask_vector <- mask
  }
  
  # Populate core components
  obj$sampling_frame <- sframe
  obj$event_table <- event_table
  obj$censor_vector <- censor_vector
  
  # Populate metadata
  obj$metadata$dataset_type <- dataset_type
  obj$metadata$TR <- TR  # Convenience copy
  obj$metadata$base_path <- normalizePath(base_path, mustWork = FALSE)
  
  # Source description
  if (dataset_type == "file_vec") {
    obj$metadata$source_description <- paste("File-based dataset with", length(images), "image files")
  } else if (dataset_type == "memory_vec") {
    obj$metadata$source_description <- paste("Memory-based dataset with", length(images), "pre-loaded objects")
  } else if (dataset_type == "matrix") {
    obj$metadata$source_description <- paste0("Matrix dataset (", nrow(images), " x ", ncol(images), ")")
  }
  
  # File options for file-based datasets
  if (dataset_type %in% c("file_vec", "bids_file")) {
    obj$metadata$file_options$mode <- image_mode
    obj$metadata$file_options$preload <- preload_data
  }
  
  # Matrix preprocessing options
  obj$metadata$matrix_options$temporal_zscore <- temporal_zscore
  obj$metadata$matrix_options$voxelwise_detrend <- voxelwise_detrend
  
  # User-provided metadata - merge directly into metadata
  for (name in names(metadata)) {
    obj$metadata[[name]] <- metadata[[name]]
  }
  
  # === VALIDATE FINAL OBJECT ===
  validate_fmri_dataset_structure(obj)
  
  # === PRELOAD DATA IF REQUESTED ===
  if (preload_data && dataset_type %in% c("file_vec", "bids_file")) {
    # This would trigger loading via get_data_matrix() when implemented
    # For now, just store the intention
    obj$metadata$file_options$preload <- TRUE
  }
  
  return(obj)
}

#' Resolve File Paths
#'
#' Internal helper to resolve relative paths against a base path.
#'
#' @param paths Character vector of file paths
#' @param base_path Base directory for relative paths
#' @return Character vector of absolute paths
#' @keywords internal
#' @noRd
resolve_paths <- function(paths, base_path) {
  if (base_path == ".") {
    # Use current working directory
    return(normalizePath(paths, mustWork = FALSE))
  } else {
    # Resolve relative to base_path
    abs_paths <- ifelse(
      is_absolute_path(paths),
      paths,
      file.path(base_path, paths)
    )
    return(normalizePath(abs_paths, mustWork = FALSE))
  }
}

#' Check if Path is Absolute
#'
#' Internal helper to determine if file paths are absolute.
#'
#' @param paths Character vector of file paths
#' @return Logical vector indicating which paths are absolute
#' @keywords internal
#' @noRd
is_absolute_path <- function(paths) {
  # Windows: starts with drive letter (C:) or UNC (\\)
  # Unix: starts with /
  grepl("^([A-Za-z]:[\\/]|[\\/]|\\\\)", paths)
} 