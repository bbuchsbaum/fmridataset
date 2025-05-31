#' Internal Utility Functions for fmridataset
#'
#' This file contains internal helper functions used throughout the fmridataset package.
#' These functions are not exported and are intended for internal use only.
#'
#' @name utils
#' @keywords internal
NULL

#' Valid dataset types recognized by the package
#'
#' @keywords internal
#' @noRd
VALID_DATASET_TYPES <- c("file_vec", "memory_vec", "matrix", "bids_file", "bids_mem")

#' Determine Dataset Type from Inputs
#'
#' Enhanced internal helper that determines the appropriate dataset_type based on 
#' the nature of the images and mask inputs. Implements refined terminology and
#' robust type detection logic.
#'
#' @param images Images input (paths, objects, or matrix)
#' @param mask Mask input (path, object, or vector) 
#' @param is_bids Logical indicating if this is from a BIDS source
#' @param preload Logical indicating if data should be preloaded
#' @return Character string indicating dataset type: 
#'   "file_vec", "memory_vec", "matrix", "bids_file", "bids_mem"
#' @keywords internal
#' @noRd
determine_dataset_type <- function(images, mask, is_bids = FALSE, preload = FALSE) {
  
  # Handle BIDS cases first (highest priority)
  if (is_bids) {
    if (preload) {
      return("bids_mem")
    } else {
      return("bids_file")
    }
  }
  
  # Enhanced type detection for non-BIDS cases
  if (is.character(images)) {
    # Character vector - should be file paths
    if (length(images) == 0) {
      stop("Images character vector is empty")
    }
    
    # Check if they look like file paths
    if (any(is.na(images) | images == "")) {
      stop("Images contain NA or empty strings")
    }
    
    # Check for common neuroimaging file extensions
    valid_extensions <- c("\\.nii$", "\\.nii\\.gz$", "\\.img$", "\\.hdr$")
    has_neuro_ext <- any(sapply(valid_extensions, function(ext) any(grepl(ext, images, ignore.case = TRUE))))
    
    if (!has_neuro_ext) {
      warning("Image files do not have common neuroimaging extensions (.nii, .nii.gz, .img, .hdr)")
    }
    
    return("file_vec")
    
  } else if (is.matrix(images) || is.array(images)) {
    # Matrix or array data
    if (is.array(images)) {
      # Validate array dimensions
      dims <- dim(images)
      if (length(dims) == 2) {
        # 2D array is fine (time x voxels)
        return("matrix")
      } else if (length(dims) == 4) {
        # 4D array might be a NeuroVol-like object, but we'll treat as matrix for now
        # Could be (x, y, z, time) - would need reshaping
        warning("4D array detected - will be treated as matrix but may need reshaping")
        return("matrix")
      } else {
        stop("Array must be 2-dimensional (time x voxels) or 4-dimensional (x x y x z x time)")
      }
    }
    
    # Check matrix dimensions are reasonable
    if (nrow(images) == 0 || ncol(images) == 0) {
      stop("Matrix has zero rows or columns")
    }
    
    return("matrix")
    
  } else if (is.list(images)) {
    # List - should contain pre-loaded neuroimaging objects
    if (length(images) == 0) {
      stop("Images list is empty")
    }
    
    # Check if any elements are NULL
    if (any(sapply(images, is.null))) {
      stop("Images list contains NULL elements")
    }
    
    # Try to detect NeuroVec/NeuroVol objects if neuroim2 is available
    if (requireNamespace("neuroim2", quietly = TRUE)) {
      # Check if elements are NeuroVec or NeuroVol objects
      is_neurovol <- sapply(images, function(x) inherits(x, "NeuroVol"))
      is_neurovec <- sapply(images, function(x) inherits(x, "NeuroVec"))
      is_neuro_obj <- is_neurovol | is_neurovec
      
      if (!all(is_neuro_obj)) {
        warning("Not all list elements appear to be NeuroVec/NeuroVol objects")
      }
    } else {
      # Without neuroim2, we can't validate the objects
      warning("Cannot validate NeuroVec/NeuroVol objects (neuroim2 not available)")
    }
    
    return("memory_vec")
    
  } else if (is.data.frame(images)) {
    stop("data.frame is not a valid images input - use matrix instead")
    
  } else if (is.vector(images) && !is.character(images)) {
    stop("Numeric/logical vectors are not valid images input - use matrix instead")
    
  } else {
    # Unknown type
    input_class <- paste(class(images), collapse = ", ")
    stop("Unable to determine dataset type from images input of class: ", input_class, 
         "\nSupported types: character (file paths), matrix/array (data), list (NeuroVec objects)")
  }
}

#' Validate Dataset Type and Input Consistency
#'
#' Validates that the determined dataset type is consistent with the actual inputs
#' and that the mask type is compatible with the images type.
#'
#' @param dataset_type Character string indicating the dataset type
#' @param images Images input 
#' @param mask Mask input
#' @return TRUE if valid, throws error if invalid
#' @keywords internal
#' @noRd
validate_dataset_type_consistency <- function(dataset_type, images, mask) {
  
  # Validate dataset_type is recognized
  valid_types <- VALID_DATASET_TYPES
  if (!dataset_type %in% valid_types) {
    stop("Invalid dataset_type: ", dataset_type, ". Must be one of: ", 
         paste(valid_types, collapse = ", "))
  }
  
  # Validate images/mask type consistency
  if (dataset_type == "file_vec") {
    if (!is.character(images)) {
      stop("dataset_type 'file_vec' requires character images input")
    }
    if (!is.null(mask) && !is.character(mask)) {
      stop("dataset_type 'file_vec' requires character or NULL mask input")
    }
    
  } else if (dataset_type == "memory_vec") {
    if (!is.list(images)) {
      stop("dataset_type 'memory_vec' requires list images input")
    }
    # Mask should be NeuroVol object or NULL, but we can't validate without neuroim2
    
  } else if (dataset_type == "matrix") {
    if (!is.matrix(images) && !is.array(images)) {
      stop("dataset_type 'matrix' requires matrix or array images input")
    }
    if (!is.null(mask) && !is.logical(mask) && !is.numeric(mask)) {
      stop("dataset_type 'matrix' requires logical/numeric vector or NULL mask input")
    }
    
  } else if (dataset_type %in% c("bids_file", "bids_mem")) {
    # BIDS types have special validation requirements that depend on bidser
    # For now, just check that images could be valid
    if (!is.character(images) && !is.list(images) && !is.matrix(images)) {
      stop("BIDS dataset_type requires character, list, or matrix images input")
    }
  }
  
  TRUE
}

#' Infer Run Lengths from Images
#'
#' Attempts to infer run lengths from image dimensions when not explicitly provided.
#' This is useful for BIDS datasets where run lengths can be read from NIfTI headers.
#'
#' @param images Images input
#' @param dataset_type Character string indicating dataset type
#' @return Integer vector of run lengths, or NULL if cannot be determined
#' @keywords internal
#' @noRd
infer_run_lengths <- function(images, dataset_type) {
  
  if (dataset_type == "matrix") {
    # For matrix, we can't infer runs - need explicit specification
    return(NULL)
    
  } else if (dataset_type == "file_vec") {
    # For files, we would need to read NIfTI headers
    # This requires neuroim2 which is in Suggests
    if (requireNamespace("neuroim2", quietly = TRUE)) {
      tryCatch({
        run_lengths <- integer(length(images))
        for (i in seq_along(images)) {
          # Read just the header to get dimensions
          img_info <- neuroim2::read_vol(images[i])
          run_lengths[i] <- dim(img_info)[4]  # 4th dimension is time
        }
        return(run_lengths)
      }, error = function(e) {
        warning("Could not read image headers to infer run lengths: ", e$message)
        return(NULL)
      })
    } else {
      warning("Cannot infer run lengths from files (neuroim2 not available)")
      return(NULL)
    }
    
  } else if (dataset_type == "memory_vec") {
    # For pre-loaded objects, we can get dimensions directly
    if (requireNamespace("neuroim2", quietly = TRUE)) {
      tryCatch({
        run_lengths <- integer(length(images))
        for (i in seq_along(images)) {
          dims <- dim(images[[i]])
          run_lengths[i] <- dims[length(dims)]  # Last dimension should be time
        }
        return(run_lengths)
      }, error = function(e) {
        warning("Could not get dimensions from pre-loaded objects: ", e$message)
        return(NULL)
      })
    } else {
      return(NULL)
    }
    
  } else {
    # BIDS types would use bidser functions
    return(NULL)
  }
}

#' Check if Package is Available
#'
#' Helper function to check if a suggested package is available and optionally
#' provide a helpful error message if it's not.
#'
#' @param pkg Character string naming the package
#' @param purpose Character string describing what the package is needed for
#' @param error Logical indicating whether to throw an error if package is not available
#' @return Logical indicating whether package is available
#' @keywords internal
#' @noRd
check_package_available <- function(pkg, purpose = NULL, error = FALSE) {
  available <- requireNamespace(pkg, quietly = TRUE)
  
  if (!available && error) {
    msg <- paste0("Package '", pkg, "' is required")
    if (!is.null(purpose)) {
      msg <- paste0(msg, " for ", purpose)
    }
    msg <- paste0(msg, " but is not installed.\nInstall with: install.packages('", pkg, "')")
    stop(msg)
  }
  
  return(available)
}

#' Validate File Extensions
#'
#' Validates that file paths have appropriate extensions for neuroimaging data.
#'
#' @param file_paths Character vector of file paths
#' @param required_extensions Character vector of allowed extensions (regex patterns)
#' @param error Logical indicating whether to throw error for invalid extensions
#' @return Logical vector indicating which paths have valid extensions
#' @keywords internal
#' @noRd
validate_file_extensions <- function(file_paths, 
                                   required_extensions = c("\\.nii$", "\\.nii\\.gz$", "\\.img$", "\\.hdr$"),
                                   error = FALSE) {
  
  valid <- rep(FALSE, length(file_paths))
  
  for (i in seq_along(file_paths)) {
    for (ext in required_extensions) {
      if (grepl(ext, file_paths[i], ignore.case = TRUE)) {
        valid[i] <- TRUE
        break
      }
    }
  }
  
  if (error && !all(valid)) {
    invalid_files <- file_paths[!valid]
    stop("Invalid file extensions detected. Expected one of: ", 
         paste(gsub("\\\\|\\$|\\^", "", required_extensions), collapse = ", "),
         "\nInvalid files: ", paste(invalid_files, collapse = ", "))
  }
  
  return(valid)
}

#' Safe File Existence Check
#'
#' Checks if files exist with better error handling and informative messages.
#'
#' @param file_paths Character vector of file paths to check
#' @param context Character string describing the context (e.g., "image files", "mask file")
#' @return Logical vector indicating which files exist
#' @keywords internal
#' @noRd
safe_file_exists <- function(file_paths, context = "files") {
  if (length(file_paths) == 0) {
    return(logical(0))
  }
  
  # Handle NA or empty strings
  valid_paths <- !is.na(file_paths) & file_paths != ""
  if (!all(valid_paths)) {
    warning("Some ", context, " are NA or empty strings")
  }
  
  # Check existence for valid paths
  exists <- rep(FALSE, length(file_paths))
  exists[valid_paths] <- file.exists(file_paths[valid_paths])

  return(exists)
}

#' Return First Non-NULL Argument
#'
#' Simple helper that returns the first argument that is not NULL.
#'
#' @param ... Arguments to check
#' @return The first non-NULL argument, or NULL if all are NULL
#' @keywords internal
#' @noRd
first_non_null <- function(...) {
  args <- list(...)
  for (arg in args) {
    if (!is.null(arg)) {
      return(arg)
    }
  }
  return(NULL)
}
