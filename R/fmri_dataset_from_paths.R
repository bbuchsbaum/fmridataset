#' Create fmri_dataset from File Paths
#'
#' This file implements the `as.fmri_dataset.character()` method for creating
#' `fmri_dataset` objects from file paths. This is one of the most common
#' use cases for creating fMRI datasets.
#'
#' @name fmri_dataset_from_paths
NULL

#' Convert Character Vector to fmri_dataset
#'
#' Creates an `fmri_dataset` object from a character vector of file paths.
#' This method handles NIfTI files on disk and validates file existence
#' and appropriate extensions.
#'
#' @param x Character vector of paths to NIfTI image files
#' @param TR Numeric scalar indicating the repetition time in seconds
#' @param run_lengths Numeric vector indicating the number of timepoints in each run.
#'   Must sum to the total number of timepoints across all images.
#' @param mask Character scalar path to NIfTI mask file, or NULL for no mask
#' @param event_table Data.frame/tibble of event data, or character path to TSV file, or NULL
#' @param censor_vector Logical or numeric vector for censoring timepoints, or NULL
#' @param base_path Character scalar base path for resolving relative paths (default: ".")
#' @param image_mode Character scalar indicating file loading mode (default: "normal")
#' @param preload_data Logical indicating whether to preload image data (default: FALSE)
#' @param temporal_zscore Logical indicating whether to apply temporal z-scoring (default: FALSE)
#' @param voxelwise_detrend Logical indicating whether to apply voxelwise detrending (default: FALSE)
#' @param metadata List of additional metadata to include
#' @param ... Additional arguments (currently unused)
#' 
#' @return An `fmri_dataset` object with dataset_type "file_vec"
#' 
#' @details
#' This method performs the following validations:
#' \itemize{
#'   \item Checks that all file paths exist
#'   \item Validates neuroimaging file extensions (.nii, .nii.gz, .img, .hdr)
#'   \item Ensures TR and run_lengths are properly specified
#'   \item Validates mask file if provided
#'   \item Processes event_table from file or data.frame
#' }
#' 
#' The `base_path` parameter is used to resolve relative paths in `x`, `mask`,
#' and `event_table` (if character). All paths are stored internally as absolute paths.
#' 
#' @examples
#' \dontrun{
#' # Create dataset from file paths
#' img_files <- c("run1.nii.gz", "run2.nii.gz")
#' mask_file <- "mask.nii.gz"
#' 
#' dataset <- as.fmri_dataset(
#'   img_files,
#'   TR = 2.0,
#'   run_lengths = c(200, 180),
#'   mask = mask_file,
#'   base_path = "/path/to/data"
#' )
#' }
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_dataset_create}} for the primary constructor
as.fmri_dataset.character <- function(x, TR, run_lengths, 
                                    mask = NULL,
                                    event_table = NULL,
                                    censor_vector = NULL,
                                    base_path = ".",
                                    image_mode = "normal",
                                    preload_data = FALSE,
                                    temporal_zscore = FALSE,
                                    voxelwise_detrend = FALSE,
                                    metadata = list(),
                                    ...) {
  
  # Validate required arguments
  if (missing(TR)) {
    stop("TR is required for file-based fmri_dataset")
  }
  if (missing(run_lengths)) {
    stop("run_lengths is required for file-based fmri_dataset")
  }
  
  # Validate that x is a character vector of file paths
  if (!is.character(x) || length(x) == 0) {
    stop("x must be a non-empty character vector of file paths")
  }
  
  # Check for NA or empty strings
  if (any(is.na(x) | x == "")) {
    stop("File paths cannot be NA or empty strings")
  }
  
  # Validate file extensions
  validate_file_extensions(x, error = TRUE)
  
  # Resolve paths relative to base_path
  resolved_paths <- resolve_paths(x, base_path)
  
  # Check file existence
  file_exists <- safe_file_exists(resolved_paths, "image files")
  if (!all(file_exists)) {
    missing_files <- resolved_paths[!file_exists]
    stop("Image files do not exist:\n", paste(missing_files, collapse = "\n"))
  }
  
  # Validate and resolve mask path if provided
  resolved_mask <- NULL
  if (!is.null(mask)) {
    if (!is.character(mask) || length(mask) != 1) {
      stop("mask must be a single character string (file path) or NULL")
    }
    if (is.na(mask) || mask == "") {
      stop("mask path cannot be NA or empty string")
    }
    
    resolved_mask <- resolve_paths(mask, base_path)
    if (!safe_file_exists(resolved_mask, "mask file")) {
      stop("Mask file does not exist: ", resolved_mask)
    }
  }
  
  # Call the primary constructor
  fmri_dataset_create(
    images = resolved_paths,
    mask = resolved_mask,
    TR = TR,
    run_lengths = run_lengths,
    event_table = event_table,
    censor_vector = censor_vector,
    base_path = base_path,
    image_mode = image_mode,
    preload_data = preload_data,
    temporal_zscore = temporal_zscore,
    voxelwise_detrend = voxelwise_detrend,
    metadata = metadata,
    ...
  )
}

#' Resolve File Paths
#'
#' Internal helper to resolve file paths relative to a base path.
#' Converts relative paths to absolute paths while preserving already-absolute paths.
#'
#' @param paths Character vector of file paths
#' @param base_path Character scalar base path for relative path resolution
#' @return Character vector of resolved absolute paths
#' @keywords internal
#' @noRd
resolve_paths <- function(paths, base_path = ".") {
  if (length(paths) == 0) {
    return(character(0))
  }
  
  # Identify which paths are already absolute
  is_absolute <- is_absolute_path(paths)
  
  # Resolve relative paths
  resolved <- character(length(paths))
  resolved[is_absolute] <- paths[is_absolute]
  resolved[!is_absolute] <- file.path(base_path, paths[!is_absolute])
  
  # Normalize paths (resolve .., ., etc.)
  normalized <- normalizePath(resolved, mustWork = FALSE)
  
  return(normalized)
}

#' Check if Path is Absolute
#'
#' Cross-platform check for absolute file paths.
#'
#' @param paths Character vector of file paths
#' @return Logical vector indicating which paths are absolute
#' @keywords internal
#' @noRd
is_absolute_path <- function(paths) {
  if (length(paths) == 0) {
    return(logical(0))
  }
  
  # On Windows, absolute paths start with drive letter (C:) or UNC (\\)
  # On Unix-like systems, absolute paths start with /
  if (.Platform$OS.type == "windows") {
    # Windows: C:\, D:\, \\server\share, etc.
    grepl("^[A-Za-z]:|^\\\\", paths)
  } else {
    # Unix-like: starts with /
    grepl("^/", paths)
  }
} 