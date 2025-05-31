#' Preloading Helper for fmri_dataset Objects
#'
#' This file implements the `preload_data()` helper function that allows explicit
#' control over preloading data into the cache. This separates side-effects from
#' constructors and provides fine-grained control over memory usage.
#'
#' @name fmri_dataset_preload
NULL

#' Preload Data into fmri_dataset Cache
#'
#' **Ticket #15**: Explicit preloading helper function with content control.
#' Separates side-effects from constructors by providing a dedicated function
#' for loading data into the internal cache.
#'
#' @param x An `fmri_dataset` object
#' @param what Character vector indicating what to preload. Options:
#'   \itemize{
#'     \item "images" - Load image data matrix
#'     \item "mask" - Load mask volume/vector
#'     \item "all" - Load both images and mask
#'   }
#' @param preprocessing Logical indicating whether to preload with preprocessing applied (default: TRUE)
#' @param runs Integer vector of specific runs to preload, or NULL for all runs (default: NULL)
#' @param force Logical indicating whether to reload if already cached (default: FALSE)
#' @param verbose Logical indicating whether to show progress messages (default: TRUE)
#' 
#' @return The input `fmri_dataset` object (invisibly), with data loaded into cache
#' 
#' @details
#' This function provides explicit control over data preloading:
#' \itemize{
#'   \item \strong{Separation of Concerns}: Keeps constructors clean by separating preloading
#'   \item \strong{Memory Management}: Allows users to control when large data is loaded
#'   \item \strong{Selective Loading}: Can preload only specific components or runs
#'   \item \strong{Cache Management}: Integrates with the internal caching system
#' }
#' 
#' **Cache Keys Created**:
#' - "raw_data_matrix" - Raw image data
#' - "data_matrix_all_preproc_TRUE/FALSE" - Processed/unprocessed full data
#' - "data_matrix_{run_ids}_preproc_TRUE/FALSE" - Run-specific data
#' - "mask_vector" / "mask_volume" - Mask data
#' 
#' @examples
#' \dontrun{
#' # Create dataset without preloading
#' dataset <- as.fmri_dataset(file_paths, TR = 2.0, run_lengths = c(200, 180))
#' 
#' # Preload everything
#' dataset <- preload_data(dataset, what = "all")
#' 
#' # Preload only images with preprocessing
#' dataset <- preload_data(dataset, what = "images", preprocessing = TRUE)
#' 
#' # Preload specific runs
#' dataset <- preload_data(dataset, what = "images", runs = c(1, 2))
#' 
#' # Preload mask only
#' dataset <- preload_data(dataset, what = "mask")
#' 
#' # Check what's cached
#' cache_summary(dataset)
#' }
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{clear_cache}}, \code{\link{cache_summary}}
preload_data <- function(x, what = c("images", "mask", "all"), 
                        preprocessing = TRUE, 
                        runs = NULL, 
                        force = FALSE, 
                        verbose = TRUE) {
  
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  what <- match.arg(what, several.ok = TRUE)
  
  # Expand "all" to specific components
  if ("all" %in% what) {
    what <- c("images", "mask")
  }
  
  # Remove duplicates
  what <- unique(what)
  
  if (verbose) {
    message("Preloading ", paste(what, collapse = " and "), " for fmri_dataset...")
  }
  
  # Preload images if requested
  if ("images" %in% what) {
    preload_images(x, preprocessing, runs, force, verbose)
  }
  
  # Preload mask if requested
  if ("mask" %in% what) {
    preload_mask(x, force, verbose)
  }
  
  if (verbose) {
    message("Preloading complete.")
  }
  
  invisible(x)
}

#' Clear Cached Data
#'
#' Removes cached data from an fmri_dataset object to free memory.
#' This is useful when working with large datasets or when you want to
#' force reloading of data.
#'
#' @param x An `fmri_dataset` object
#' @param what Character vector specifying what to clear. Options:
#'   - "all" (default): Clear all cached data
#'   - "images": Clear cached image data matrices
#'   - "mask": Clear cached mask data
#'   - Specific cache keys
#' @param verbose Logical, whether to show messages about what was cleared
#' @return The fmri_dataset object (invisibly)
#' 
#' @export
#' @family fmri_dataset
clear_cache.fmri_dataset <- function(x, what = "all", verbose = TRUE, ...) {
  
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  if ("all" %in% what) {
    # Clear everything
    cached_objects <- ls(x$data_cache)
    if (length(cached_objects) > 0) {
      rm(list = cached_objects, envir = x$data_cache)
      if (verbose) {
        message("Cleared all cached data (", length(cached_objects), " objects)")
      }
    } else {
      if (verbose) {
        message("No cached data to clear")
      }
    }
    
  } else {
    # Clear specific items
    cached_objects <- ls(x$data_cache)
    
    for (item in what) {
      if (item == "images") {
        # Clear all image-related cache
        image_keys <- grep("^(raw_)?data_matrix", cached_objects, value = TRUE)
        if (length(image_keys) > 0) {
          rm(list = image_keys, envir = x$data_cache)
          if (verbose) {
            message("Cleared image cache (", length(image_keys), " objects)")
          }
        }
        
      } else if (item == "mask") {
        # Clear mask cache
        mask_keys <- grep("^mask_", cached_objects, value = TRUE)
        if (length(mask_keys) > 0) {
          rm(list = mask_keys, envir = x$data_cache)
          if (verbose) {
            message("Cleared mask cache (", length(mask_keys), " objects)")
          }
        }
        
      } else if (item %in% cached_objects) {
        # Clear specific cache key
        rm(list = item, envir = x$data_cache)
        if (verbose) {
          message("Cleared cache key: ", item)
        }
        
      } else {
        if (verbose) {
          warning("Cache key not found: ", item)
        }
      }
    }
  }
  
  invisible(x)
}

#' Cache Summary
#'
#' Provides information about cached data in an fmri_dataset object.
#'
#' @param x An `fmri_dataset` object
#' @return A data.frame with cache information
#' 
#' @export
#' @family fmri_dataset
cache_summary <- function(x) {
  
  if (!is.fmri_dataset(x)) {
    stop("x must be an fmri_dataset object")
  }
  
  cached_objects <- ls(x$data_cache)
  
  if (length(cached_objects) == 0) {
    message("No cached data")
    return(invisible(NULL))
  }
  
  # Get sizes and types
  cache_info <- data.frame(
    cache_key = cached_objects,
    object_class = sapply(cached_objects, function(key) {
      obj <- get(key, envir = x$data_cache)
      paste(class(obj), collapse = ", ")
    }),
    size_bytes = sapply(cached_objects, function(key) {
      obj <- get(key, envir = x$data_cache)
      as.numeric(object.size(obj))
    }),
    stringsAsFactors = FALSE
  )
  
  # Add human-readable sizes
  cache_info$size_mb <- round(cache_info$size_bytes / (1024^2), 2)
  
  # Sort by size
  cache_info <- cache_info[order(cache_info$size_bytes, decreasing = TRUE), ]
  rownames(cache_info) <- NULL
  
  total_mb <- round(sum(cache_info$size_bytes) / (1024^2), 2)
  
  message("Cache Summary (Total: ", total_mb, " MB)")
  print(cache_info[, c("cache_key", "object_class", "size_mb")])
  
  invisible(cache_info)
}

# ============================================================================
# Internal Helper Functions for Preloading
# ============================================================================

#' Preload Images
#'
#' Internal helper to preload image data.
#'
#' @param x fmri_dataset object
#' @param preprocessing Logical for preprocessing
#' @param runs Run IDs or NULL
#' @param force Force reload
#' @param verbose Show messages
#' @keywords internal
#' @noRd
preload_images <- function(x, preprocessing, runs, force, verbose) {
  
  dataset_type <- x$metadata$dataset_type
  
  # Skip if already in memory
  if (dataset_type == "matrix") {
    if (verbose) {
      message("  Images: Already in memory (matrix dataset)")
    }
    return(invisible(NULL))
  }
  
  if (dataset_type == "memory_vec" && !force) {
    if (verbose) {
      message("  Images: Already in memory (pre-loaded objects)")
    }
    return(invisible(NULL))
  }
  
  # For file-based datasets, trigger loading
  if (dataset_type %in% c("file_vec", "bids_file", "bids_mem")) {
    
    if (is.null(runs)) {
      # Load all data
      if (verbose) {
        message("  Images: Loading all runs...")
      }
      
      # This will trigger caching in get_data_matrix
      data_matrix <- get_data_matrix(x, 
                                   run_id = NULL, 
                                   apply_preprocessing = preprocessing, 
                                   force_reload = force)
      
      if (verbose) {
        message("    Loaded ", nrow(data_matrix), " timepoints x ", 
                ncol(data_matrix), " voxels")
      }
      
    } else {
      # Load specific runs
      if (verbose) {
        message("  Images: Loading runs ", paste(runs, collapse = ", "), "...")
      }
      
      data_matrix <- get_data_matrix(x, 
                                   run_id = runs, 
                                   apply_preprocessing = preprocessing, 
                                   force_reload = force)
      
      if (verbose) {
        message("    Loaded ", nrow(data_matrix), " timepoints x ", 
                ncol(data_matrix), " voxels")
      }
    }
  }
}

#' Preload Mask
#'
#' Internal helper to preload mask data.
#'
#' @param x fmri_dataset object  
#' @param force Force reload
#' @param verbose Show messages
#' @keywords internal
#' @noRd
preload_mask <- function(x, force, verbose) {
  
  # Check if mask exists
  if (is.null(x$mask_path) && is.null(x$mask_object) && is.null(x$mask_vector)) {
    if (verbose) {
      message("  Mask: No mask to preload")
    }
    return(invisible(NULL))
  }
  
  # Skip if already in memory (for matrix and memory_vec)
  dataset_type <- x$metadata$dataset_type
  if (dataset_type %in% c("matrix", "memory_vec") && !force) {
    if (verbose) {
      message("  Mask: Already in memory")
    }
    return(invisible(NULL))
  }
  
  # Load mask (both vector and volume forms)
  if (verbose) {
    message("  Mask: Loading...")
  }
  
  # Load as vector
  mask_vector <- get_mask_volume(x, as_vector = TRUE, force_reload = force)
  
  # Load as volume (if possible)
  if (!is.null(x$mask_path) || !is.null(x$mask_object)) {
    mask_volume <- get_mask_volume(x, as_vector = FALSE, force_reload = force)
  }
  
  if (verbose && !is.null(mask_vector)) {
    message("    Loaded mask with ", sum(mask_vector), " / ", length(mask_vector), " voxels")
  }
} 