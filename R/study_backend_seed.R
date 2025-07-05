#' StudyBackendSeed for DelayedArray
#'
#' A DelayedArray seed that provides lazy access to multi-subject fMRI data
#' without loading all subjects into memory at once.
#'
#' @import methods
#' @import DelayedArray
#' @importFrom methods new setClass setMethod
#' @importClassesFrom DelayedArray DelayedArray
#' @importFrom DelayedArray RegularArrayGrid is_sparse
#' @keywords internal
setClass("StudyBackendSeed",
  slots = c(
    backends = "list",           # List of subject backends
    subject_ids = "character",   # Subject identifiers
    subject_boundaries = "integer", # Row indices where subjects start
    dim = "integer",            # Total dimensions (time x voxels) - note: singular "dim"
    cache = "environment"       # Cache for loaded chunks
  )
)

#' Constructor for StudyBackendSeed
#' @keywords internal
StudyBackendSeed <- function(backends, subject_ids) {
  # Get dimensions from each backend
  dims_list <- lapply(backends, backend_get_dims)
  time_dims <- vapply(dims_list, function(x) x$time, integer(1))
  spatial_dims <- dims_list[[1]]$spatial
  n_voxels <- prod(spatial_dims)
  
  # Calculate subject boundaries (where each subject's data starts)
  subject_boundaries <- c(0L, cumsum(time_dims))
  
  # Total dimensions
  total_time <- sum(time_dims)
  dims <- c(total_time, n_voxels)
  
  # Create cache environment with size limit
  cache_size_mb <- getOption("fmridataset.study_cache_mb", 1024)
  cache <- new.env(parent = emptyenv())
  cache$max_size <- cache_size_mb * 1024^2
  cache$current_size <- 0
  cache$lru <- list()  # Simple LRU tracking
  
  new("StudyBackendSeed",
    backends = backends,
    subject_ids = as.character(subject_ids),
    subject_boundaries = as.integer(subject_boundaries),
    dim = as.integer(dims),
    cache = cache
  )
}

#' Dimensions of StudyBackendSeed
#' 
#' @param x A StudyBackendSeed object
#' @return Integer vector of dimensions
#' @rdname dim-StudyBackendSeed-method
#' @aliases dim,StudyBackendSeed-method
#' @keywords internal
setMethod("dim", "StudyBackendSeed", function(x) x@dim)

#' Dimnames of StudyBackendSeed
#' 
#' @param x A StudyBackendSeed object
#' @return List of dimnames (always NULL for this class)
#' @rdname dimnames-StudyBackendSeed-method  
#' @aliases dimnames,StudyBackendSeed-method
#' @keywords internal
setMethod("dimnames", "StudyBackendSeed", function(x) {
  list(NULL, NULL)  # No dimnames by default
})

#' Extract array subset from StudyBackendSeed
#'
#' This is the key method that enables lazy evaluation. It only loads
#' the specific subjects and voxels requested.
#'
#' @rdname extract_array
#' @aliases extract_array,StudyBackendSeed-method
#' @importFrom S4Arrays extract_array
setMethod("extract_array", "StudyBackendSeed", function(x, index) {
  # Handle the index properly
  if (!is.list(index)) {
    stop("index must be a list")
  }
  
  # Get row and column indices
  row_idx <- index[[1]]
  col_idx <- index[[2]]
  
  # Convert NULL to full range
  if (is.null(row_idx)) row_idx <- seq_len(x@dim[1])
  if (is.null(col_idx)) col_idx <- seq_len(x@dim[2])
  
  # Convert logical to integer indices
  if (is.logical(row_idx)) row_idx <- which(row_idx)
  if (is.logical(col_idx)) col_idx <- which(col_idx)
  
  # Determine which subjects we need
  subjects_needed <- find_subjects_for_rows(row_idx, x@subject_boundaries)
  
  # Collect data from each needed subject
  result_rows <- list()
  
  for (subj_idx in subjects_needed) {
    # Calculate which rows from this subject we need
    subj_start <- x@subject_boundaries[subj_idx] + 1L
    subj_end <- x@subject_boundaries[subj_idx + 1]
    subj_rows <- seq(subj_start, subj_end)
    
    # Find intersection with requested rows
    rows_to_get <- intersect(row_idx, subj_rows)
    if (length(rows_to_get) == 0) next
    
    # Convert to subject-local indices
    local_rows <- rows_to_get - x@subject_boundaries[subj_idx]
    
    # Check cache first
    cache_key <- paste0("subj_", subj_idx, "_cols_", paste(range(col_idx), collapse = "_"))
    
    if (exists(cache_key, envir = x@cache)) {
      # Use cached data
      subj_data <- get(cache_key, envir = x@cache)
      result_rows[[length(result_rows) + 1]] <- subj_data[local_rows, , drop = FALSE]
    } else {
      # Load from backend
      backend <- x@backends[[subj_idx]]
      subj_data <- backend_get_data(backend, rows = NULL, cols = col_idx)
      
      # Convert to regular matrix if it's a DelayedArray
      if (is(subj_data, "DelayedArray")) {
        subj_data <- as.matrix(subj_data)
      }
      
      # Cache if not too large
      data_size <- object.size(subj_data)
      if (data_size < x@cache$max_size / 10) {  # Don't cache if > 10% of cache size
        assign(cache_key, subj_data, envir = x@cache)
        x@cache$current_size <- x@cache$current_size + as.numeric(data_size)
        
        # Simple cache eviction if needed
        if (x@cache$current_size > x@cache$max_size) {
          evict_cache_entries(x@cache)
        }
      }
      
      result_rows[[length(result_rows) + 1]] <- subj_data[local_rows, , drop = FALSE]
    }
  }
  
  # Combine results
  if (length(result_rows) == 0) {
    matrix(numeric(0), nrow = 0, ncol = length(col_idx))
  } else {
    do.call(rbind, result_rows)
  }
})

#' Find which subjects contain the requested rows
#' @keywords internal
find_subjects_for_rows <- function(rows, boundaries) {
  subjects <- integer()
  for (i in seq_len(length(boundaries) - 1)) {
    subj_start <- boundaries[i] + 1L
    subj_end <- boundaries[i + 1]
    if (any(rows >= subj_start & rows <= subj_end)) {
      subjects <- c(subjects, i)
    }
  }
  subjects
}

#' Simple cache eviction
#' @keywords internal  
evict_cache_entries <- function(cache_env) {
  # Remove oldest entries until we're under the limit
  # This is a simple implementation - could be improved with proper LRU
  all_keys <- ls(cache_env)
  if (length(all_keys) > 0) {
    # Remove first half of entries
    to_remove <- all_keys[1:(length(all_keys) %/% 2)]
    rm(list = to_remove, envir = cache_env)
    cache_env$current_size <- cache_env$current_size / 2  # Approximate
  }
}

#' Check if object is sparse (always FALSE for fMRI data)
#' @rdname is_sparse
#' @aliases is_sparse,StudyBackendSeed-method
setMethod("is_sparse", "StudyBackendSeed", function(x) FALSE)

# Note: chunkGrid method omitted as it requires matching the exact generic signature
# DelayedArray will use default chunking which is fine for our use case