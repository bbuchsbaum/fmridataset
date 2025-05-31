#' Create fmri_dataset from Lists and Matrices
#'
#' This file implements the `as.fmri_dataset.list()` and `as.fmri_dataset.matrix()` 
#' methods for creating `fmri_dataset` objects from pre-loaded NeuroVec objects
#' and in-memory data matrices.
#'
#' @name fmri_dataset_from_list_matrix
NULL

#' Convert List to fmri_dataset
#'
#' Creates an `fmri_dataset` object from a list of pre-loaded NeuroVec/NeuroVol objects.
#' This method is useful when you have already loaded neuroimaging data into memory
#' and want to create a unified dataset interface.
#'
#' @param x List containing NeuroVec or NeuroVol objects (one per run)
#' @param TR Numeric scalar indicating the repetition time in seconds
#' @param run_lengths Numeric vector indicating the number of timepoints in each run.
#'   Must match the temporal dimensions of objects in `x`.
#' @param mask NeuroVol mask object, or NULL for no mask
#' @param event_table Data.frame/tibble of event data, or character path to TSV file, or NULL
#' @param censor_vector Logical or numeric vector for censoring timepoints, or NULL
#' @param temporal_zscore Logical indicating whether to apply temporal z-scoring (default: FALSE)
#' @param voxelwise_detrend Logical indicating whether to apply voxelwise detrending (default: FALSE)
#' @param metadata List of additional metadata to include
#' @param ... Additional arguments (currently unused)
#' 
#' @return An `fmri_dataset` object with dataset_type "memory_vec"
#' 
#' @details
#' This method performs the following validations:
#' \itemize{
#'   \item Checks that all list elements are NeuroVec/NeuroVol objects (if neuroim2 available)
#'   \item Validates that object dimensions match run_lengths
#'   \item Ensures spatial dimensions are consistent across runs
#'   \item Validates mask compatibility if provided
#' }
#' 
#' The `neuroim2` package is required for full validation of NeuroVec objects,
#' but the method will work with a warning if `neuroim2` is not available.
#' 
#' @examples
#' \dontrun{
#' # Assuming you have pre-loaded NeuroVec objects
#' library(neuroim2)
#' vol1 <- read_vol("run1.nii.gz")
#' vol2 <- read_vol("run2.nii.gz")
#' mask_vol <- read_vol("mask.nii.gz")
#' 
#' dataset <- as.fmri_dataset(
#'   list(vol1, vol2),
#'   TR = 2.0,
#'   run_lengths = c(200, 180),
#'   mask = mask_vol
#' )
#' }
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_dataset_create}} for the primary constructor
as.fmri_dataset.list <- function(x, TR, run_lengths,
                                mask = NULL,
                                event_table = NULL,
                                censor_vector = NULL,
                                temporal_zscore = FALSE,
                                voxelwise_detrend = FALSE,
                                metadata = list(),
                                ...) {
  
  # Validate required arguments
  if (missing(TR)) {
    stop("TR is required for list-based fmri_dataset")
  }
  if (missing(run_lengths)) {
    stop("run_lengths is required for list-based fmri_dataset")
  }
  
  # Validate that x is a non-empty list
  if (!is.list(x) || length(x) == 0) {
    stop("x must be a non-empty list of NeuroVec/NeuroVol objects")
  }
  
  # Check for NULL elements
  if (any(sapply(x, is.null))) {
    stop("List contains NULL elements")
  }
  
  # Validate run_lengths matches list length
  if (length(run_lengths) != length(x)) {
    stop("Length of run_lengths (", length(run_lengths), 
         ") must match length of object list (", length(x), ")")
  }
  
  # Enhanced validation if neuroim2 is available
  if (check_package_available("neuroim2", "validating NeuroVec objects", error = FALSE)) {
    
    # Check that all elements are NeuroVec or NeuroVol objects
    valid_objects <- sapply(x, function(obj) {
      inherits(obj, "NeuroVec") || inherits(obj, "NeuroVol")
    })
    
    if (!all(valid_objects)) {
      invalid_indices <- which(!valid_objects)
      stop("List elements at positions ", paste(invalid_indices, collapse = ", "),
           " are not NeuroVec or NeuroVol objects")
    }
    
    # Validate temporal dimensions match run_lengths
    actual_dims <- sapply(x, function(obj) {
      dims <- dim(obj)
      dims[length(dims)]  # Last dimension should be time
    })
    
    if (!all(actual_dims == run_lengths)) {
      mismatch_idx <- which(actual_dims != run_lengths)
      stop("Temporal dimensions mismatch for objects at positions ",
           paste(mismatch_idx, collapse = ", "), ":\n",
           "Expected: ", paste(run_lengths[mismatch_idx], collapse = ", "), "\n",
           "Actual: ", paste(actual_dims[mismatch_idx], collapse = ", "))
    }
    
    # Check spatial dimensions are consistent
    spatial_dims <- lapply(x, function(obj) {
      dims <- dim(obj)
      dims[-length(dims)]  # All but last dimension
    })
    
    first_spatial <- spatial_dims[[1]]
    consistent_spatial <- sapply(spatial_dims, function(dims) {
      length(dims) == length(first_spatial) && all(dims == first_spatial)
    })
    
    if (!all(consistent_spatial)) {
      inconsistent_idx <- which(!consistent_spatial)
      stop("Spatial dimensions are inconsistent for objects at positions ",
           paste(inconsistent_idx, collapse = ", "))
    }
    
    # Validate mask if provided
    if (!is.null(mask)) {
      if (!inherits(mask, "NeuroVol")) {
        stop("mask must be a NeuroVol object or NULL when using pre-loaded objects")
      }
      
      # Check mask spatial dimensions match data
      mask_dims <- dim(mask)
      if (!all(mask_dims == first_spatial)) {
        stop("Mask spatial dimensions (", paste(mask_dims, collapse = " x "), 
             ") do not match data spatial dimensions (", 
             paste(first_spatial, collapse = " x "), ")")
      }
    }
    
  } else {
    # Without neuroim2, provide warning and basic validation
    warning("neuroim2 package not available - limited validation of NeuroVec objects")
    
    # Basic check that objects aren't obviously wrong types
    basic_valid <- sapply(x, function(obj) {
      # Should be some kind of array-like object with dimensions
      !is.null(dim(obj)) && length(dim(obj)) >= 2
    })
    
    if (!all(basic_valid)) {
      invalid_indices <- which(!basic_valid)
      stop("List elements at positions ", paste(invalid_indices, collapse = ", "),
           " do not appear to be multi-dimensional array objects")
    }
  }
  
  # Call the primary constructor
  fmri_dataset_create(
    images = x,
    mask = mask,
    TR = TR,
    run_lengths = run_lengths,
    event_table = event_table,
    censor_vector = censor_vector,
    base_path = ".",  # Not applicable for pre-loaded objects
    image_mode = "normal",  # Not applicable for pre-loaded objects
    preload_data = TRUE,  # Already preloaded
    temporal_zscore = temporal_zscore,
    voxelwise_detrend = voxelwise_detrend,
    metadata = metadata,
    ...
  )
}

#' Convert Matrix to fmri_dataset
#'
#' Creates an `fmri_dataset` object from an in-memory data matrix.
#' This method is useful when you have already preprocessed fMRI data in matrix form
#' and want to create a unified dataset interface.
#'
#' @param x Numeric matrix with timepoints as rows and voxels as columns (time x voxels),
#'   or 4D array (x x y x z x time) that will be reshaped to matrix form
#' @param TR Numeric scalar indicating the repetition time in seconds
#' @param run_lengths Numeric vector indicating the number of timepoints in each run.
#'   Must sum to the total number of rows in `x`.
#' @param mask Logical vector indicating which voxels to include, logical matrix
#'   matching the spatial dimensions of `x`, or NULL for no mask
#' @param event_table Data.frame/tibble of event data, or character path to TSV file, or NULL
#' @param censor_vector Logical or numeric vector for censoring timepoints, or NULL
#' @param temporal_zscore Logical indicating whether to apply temporal z-scoring (default: FALSE)
#' @param voxelwise_detrend Logical indicating whether to apply voxelwise detrending (default: FALSE)
#' @param metadata List of additional metadata to include
#' @param ... Additional arguments (currently unused)
#' 
#' @return An `fmri_dataset` object with dataset_type "matrix"
#' 
#' @details
#' This method performs the following validations:
#' \itemize{
#'   \item Checks that matrix dimensions are reasonable (time x voxels)
#'   \item Validates that sum of run_lengths matches number of timepoints
#'   \item Ensures mask dimensions are compatible with data
#'   \item Handles 4D arrays by reshaping to 2D matrix form
#' }
#' 
#' For 4D arrays, the spatial dimensions (x, y, z) are flattened to create a
#' time x voxels matrix. The mask should either be a logical vector of length
#' equal to the number of voxels, or a 3D logical array matching the spatial
#' dimensions.
#' 
#' @examples
#' \dontrun{
#' # Create dataset from matrix
#' data_matrix <- matrix(rnorm(380 * 1000), nrow = 380, ncol = 1000)
#' mask_vec <- rep(TRUE, 1000)
#' 
#' dataset <- as.fmri_dataset(
#'   data_matrix,
#'   TR = 2.0,
#'   run_lengths = c(200, 180),
#'   mask = mask_vec
#' )
#' 
#' # Create dataset from 4D array
#' data_array <- array(rnorm(64 * 64 * 30 * 200), dim = c(64, 64, 30, 200))
#' mask_array <- array(TRUE, dim = c(64, 64, 30))
#' 
#' dataset <- as.fmri_dataset(
#'   data_array,
#'   TR = 2.0,
#'   run_lengths = 200,
#'   mask = mask_array
#' )
#' }
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_dataset_create}} for the primary constructor
as.fmri_dataset.matrix <- function(x, TR, run_lengths,
                                  mask = NULL,
                                  event_table = NULL,
                                  censor_vector = NULL,
                                  temporal_zscore = FALSE,
                                  voxelwise_detrend = FALSE,
                                  metadata = list(),
                                  ...) {
  
  # Validate required arguments
  if (missing(TR)) {
    stop("TR is required for matrix-based fmri_dataset")
  }
  if (missing(run_lengths)) {
    stop("run_lengths is required for matrix-based fmri_dataset")
  }
  
  # Validate and process matrix/array input
  data_matrix <- validate_and_process_matrix(x)
  
  # Validate run_lengths consistency
  total_timepoints <- sum(run_lengths)
  if (nrow(data_matrix) != total_timepoints) {
    stop("Sum of run_lengths (", total_timepoints, 
         ") must equal number of timepoints in matrix (", nrow(data_matrix), ")")
  }
  
  # Validate mask if provided
  processed_mask <- NULL
  if (!is.null(mask)) {
    processed_mask <- validate_and_process_mask(mask, data_matrix, x)
  }
  
  # Call the primary constructor
  fmri_dataset_create(
    images = data_matrix,
    mask = processed_mask,
    TR = TR,
    run_lengths = run_lengths,
    event_table = event_table,
    censor_vector = censor_vector,
    base_path = ".",  # Not applicable for matrices
    image_mode = "normal",  # Not applicable for matrices
    preload_data = TRUE,  # Already in memory
    temporal_zscore = temporal_zscore,
    voxelwise_detrend = voxelwise_detrend,
    metadata = metadata,
    ...
  )
}

#' Validate and Process Matrix Input
#'
#' Internal helper to validate and process matrix/array input for as.fmri_dataset.matrix
#'
#' @param x Matrix or array input
#' @return Processed matrix (time x voxels)
#' @keywords internal
#' @noRd
validate_and_process_matrix <- function(x) {
  
  if (is.matrix(x)) {
    # Already a matrix - validate dimensions
    if (nrow(x) == 0 || ncol(x) == 0) {
      stop("Matrix cannot have zero rows or columns")
    }
    
    if (!is.numeric(x)) {
      stop("Matrix must be numeric")
    }
    
    return(x)
    
  } else if (is.array(x)) {
    # Array - need to check dimensions and reshape
    dims <- dim(x)
    
    if (length(dims) == 2) {
      # 2D array is just a matrix
      return(as.matrix(x))
      
    } else if (length(dims) == 4) {
      # 4D array (x, y, z, time) - reshape to (time, voxels)
      if (any(dims == 0)) {
        stop("4D array cannot have zero-length dimensions")
      }
      
      # Time dimension is last
      n_timepoints <- dims[4]
      n_voxels <- prod(dims[1:3])
      
      # Reshape: flatten spatial dimensions, transpose to time x voxels
      reshaped <- array(x, dim = c(n_voxels, n_timepoints))
      data_matrix <- t(reshaped)  # transpose to time x voxels
      
      if (!is.numeric(data_matrix)) {
        stop("Array must be numeric")
      }
      
      return(data_matrix)
      
    } else {
      stop("Array must be 2-dimensional (time x voxels) or 4-dimensional (x x y x z x time), ",
           "got ", length(dims), " dimensions")
    }
    
  } else {
    stop("Input must be a matrix or array, got class: ", class(x)[1])
  }
}

#' Validate and Process Mask for Matrix Input
#'
#' Internal helper to validate and process mask input for matrix-based datasets
#'
#' @param mask Mask input (logical vector, logical matrix/array, or NULL)
#' @param data_matrix Processed data matrix (time x voxels)
#' @param original_x Original input (for dimension checking if 4D array)
#' @return Processed mask vector or NULL
#' @keywords internal
#' @noRd
validate_and_process_mask <- function(mask, data_matrix, original_x) {
  
  if (is.vector(mask) && is.logical(mask)) {
    # Logical vector - should match number of voxels
    if (length(mask) != ncol(data_matrix)) {
      stop("Mask vector length (", length(mask), 
           ") must match number of voxels (", ncol(data_matrix), ")")
    }
    return(mask)
    
  } else if (is.vector(mask) && is.numeric(mask)) {
    # Numeric vector - convert to logical, should be 0/1
    if (!all(mask %in% c(0, 1))) {
      stop("Numeric mask vector must contain only 0s and 1s")
    }
    logical_mask <- as.logical(mask)
    if (length(logical_mask) != ncol(data_matrix)) {
      stop("Mask vector length (", length(logical_mask), 
           ") must match number of voxels (", ncol(data_matrix), ")")
    }
    return(logical_mask)
    
  } else if (is.array(mask) || is.matrix(mask)) {
    # Array/matrix mask
    if (!is.logical(mask) && !is.numeric(mask)) {
      stop("Mask array/matrix must be logical or numeric")
    }
    
    # If original input was 4D array, mask should be 3D
    if (is.array(original_x) && length(dim(original_x)) == 4) {
      original_spatial <- dim(original_x)[1:3]
      mask_dims <- dim(mask)
      
      if (length(mask_dims) != 3 || !all(mask_dims == original_spatial)) {
        stop("Mask array dimensions (", paste(mask_dims, collapse = " x "), 
             ") must match spatial dimensions of data (", 
             paste(original_spatial, collapse = " x "), ")")
      }
      
      # Flatten mask to vector
      mask_vector <- as.vector(mask)
      if (is.numeric(mask_vector)) {
        if (!all(mask_vector %in% c(0, 1))) {
          stop("Numeric mask must contain only 0s and 1s")
        }
        mask_vector <- as.logical(mask_vector)
      }
      
      return(mask_vector)
      
    } else {
      # For 2D input, mask could be 2D but this is unusual
      warning("Array/matrix mask for 2D input - flattening to vector")
      mask_vector <- as.vector(mask)
      if (is.numeric(mask_vector)) {
        if (!all(mask_vector %in% c(0, 1))) {
          stop("Numeric mask must contain only 0s and 1s")
        }
        mask_vector <- as.logical(mask_vector)
      }
      
      if (length(mask_vector) != ncol(data_matrix)) {
        stop("Flattened mask length (", length(mask_vector), 
             ") must match number of voxels (", ncol(data_matrix), ")")
      }
      
      return(mask_vector)
    }
    
  } else {
    stop("Mask must be logical vector, numeric vector (0/1), logical array, or NULL")
  }
} 