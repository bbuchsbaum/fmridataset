#' Study Backend Seed for DelayedArray (S3 Implementation)
#'
#' @description
#' A DelayedArray-compatible seed that provides lazy access to multi-subject fMRI data
#' without loading all subjects into memory at once. This is a pure S3 implementation
#' that maintains compatibility with DelayedArray through S3 methods.
#'
#' @details
#' This class implements the DelayedArray seed contract using S3 methods, providing:
#' - Lazy loading of subject data
#' - LRU caching with configurable memory limits
#' - Efficient subset extraction
#' - Compatibility with DelayedArray operations
#'
#' @name study-backend-seed-s3
#' @importFrom utils object.size
#' @keywords internal
NULL

#' Create a Study Backend Seed
#'
#' @param backends List of subject backend objects
#' @param subject_ids Character vector of subject identifiers
#' @return A study_backend_seed S3 object
#' @keywords internal
study_backend_seed <- function(backends, subject_ids) {
  # Validate inputs
  if (!is.list(backends)) {
    stop("backends must be a list")
  }
  if (length(backends) != length(subject_ids)) {
    stop("Number of backends must match number of subject IDs")
  }

  # Get dimensions from each backend
  dims_list <- lapply(backends, backend_get_dims)
  time_dims <- vapply(dims_list, function(x) x$time, integer(1))
  spatial_dims <- dims_list[[1]]$spatial
  n_voxels <- prod(spatial_dims)

  # Validate all backends have same spatial dimensions
  for (i in seq_along(dims_list)) {
    if (!identical(dims_list[[i]]$spatial, spatial_dims)) {
      stop(sprintf("Backend %d has inconsistent spatial dimensions", i))
    }
  }

  # Calculate subject boundaries (where each subject's data starts)
  subject_boundaries <- c(0L, cumsum(time_dims))

  # Total dimensions
  total_time <- sum(time_dims)
  dims <- c(total_time, n_voxels)

  # Create LRU cache
  cache <- create_study_cache()

  structure(
    list(
      backends = backends,
      subject_ids = as.character(subject_ids),
      subject_boundaries = as.integer(subject_boundaries),
      dims = as.integer(dims),
      cache = cache
    ),
    class = "study_backend_seed"
  )
}

#' Create LRU Cache for Study Backend
#' @keywords internal
create_study_cache <- function() {
  cache_size_mb <- getOption("fmridataset.study_cache_mb", 1024)
  max_size <- cache_size_mb * 1024^2

  # Use cachem (which is in Imports)
  cachem::cache_mem(max_size = max_size)
}

#' @export
dim.study_backend_seed <- function(x) {
  x$dims
}

#' @export
dimnames.study_backend_seed <- function(x) {
  list(NULL, NULL) # No dimnames by default
}

#' Extract Array from Study Backend Seed
#'
#' @description
#' Core method for DelayedArray compatibility. Extracts the requested subset
#' of data, loading only the necessary subjects.
#'
#' @param x A study_backend_seed object
#' @param index A list with two elements: row indices and column indices
#' @return A matrix with the requested data
#' @export
extract_array.study_backend_seed <- function(x, index) {
  # Validate index
  if (!is.list(index) || length(index) != 2) {
    stop("index must be a list of length 2")
  }

  # Get row and column indices
  row_idx <- index[[1]]
  col_idx <- index[[2]]

  # Convert NULL to full range
  if (is.null(row_idx)) row_idx <- seq_len(x$dims[1])
  if (is.null(col_idx)) col_idx <- seq_len(x$dims[2])

  # Convert logical to integer indices
  if (is.logical(row_idx)) row_idx <- which(row_idx)
  if (is.logical(col_idx)) col_idx <- which(col_idx)

  # Validate indices
  if (any(row_idx < 1 | row_idx > x$dims[1])) {
    stop("Row indices out of bounds")
  }
  if (any(col_idx < 1 | col_idx > x$dims[2])) {
    stop("Column indices out of bounds")
  }

  # Determine which subjects we need
  subjects_needed <- find_subjects_for_rows(row_idx, x$subject_boundaries)

  # Pre-allocate result matrix
  result <- matrix(NA_real_, length(row_idx), length(col_idx))
  result_row_idx <- 1L

  for (subj_idx in subjects_needed) {
    # Calculate which rows from this subject we need
    subj_start <- x$subject_boundaries[subj_idx] + 1L
    subj_end <- x$subject_boundaries[subj_idx + 1]
    subj_rows <- seq(subj_start, subj_end)

    # Find intersection with requested rows
    rows_to_get <- intersect(row_idx, subj_rows)
    if (length(rows_to_get) == 0) next

    # Convert to subject-local indices
    local_rows <- rows_to_get - x$subject_boundaries[subj_idx]

    # Create cache key
    cache_key <- paste0(
      "subj_", subj_idx, "_cols_",
      paste(range(col_idx), collapse = "_")
    )

    # Try to get from cache
    subj_data <- NULL
    if (inherits(x$cache, "cachem")) {
      subj_data <- x$cache$get(cache_key)
    } else if (x$cache$exists(cache_key)) {
      subj_data <- x$cache$get(cache_key)
    }

    if (is.null(subj_data)) {
      # Load from backend
      backend <- x$backends[[subj_idx]]
      subj_data <- backend_get_data(backend, rows = NULL, cols = col_idx)

      # Convert to regular matrix if needed
      if (inherits(subj_data, "DelayedArray")) {
        subj_data <- as.matrix(subj_data)
      }

      # Cache if not too large
      data_size <- as.numeric(object.size(subj_data))
      cache_threshold <- if (inherits(x$cache, "cachem")) {
        x$cache$info()$max_size / 10
      } else {
        x$cache$max_size / 10
      }

      if (data_size < cache_threshold) {
        if (inherits(x$cache, "cachem")) {
          x$cache$set(cache_key, subj_data)
        } else {
          x$cache$set(cache_key, subj_data)
        }
      }
    }

    # Extract requested rows and place in result
    n_rows <- length(rows_to_get)
    result[result_row_idx:(result_row_idx + n_rows - 1), ] <-
      subj_data[local_rows, , drop = FALSE]
    result_row_idx <- result_row_idx + n_rows
  }

  result
}

#' Find Which Subjects Contain Given Rows
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
  unique(subjects)
}

#' Check if Study Backend Seed is Sparse
#'
#' @param x A study_backend_seed object
#' @return Logical indicating if the data is sparse
#' @export
is_sparse.study_backend_seed <- function(x) {
  FALSE # fMRI data is typically dense
}

#' Get Chunk Grid for Study Backend Seed
#'
#' @param x A study_backend_seed object
#' @param chunk_dim Optional chunk dimensions
#' @return A RegularArrayGrid object
#' @export
chunkGrid.study_backend_seed <- function(x, chunk_dim = NULL) {
  if (is.null(chunk_dim)) {
    # Default: chunk by subject
    chunk_dim <- c(
      min(x$subject_boundaries[2] - x$subject_boundaries[1], x$dims[1]),
      x$dims[2]
    )
  }

  DelayedArray::RegularArrayGrid(
    refdim = x$dims,
    spacings = chunk_dim
  )
}

#' Register S3 Methods for DelayedArray Compatibility
#'
#' @description
#' This function should be called in .onLoad to register the S3 methods
#' with DelayedArray's S4 generic functions.
#'
#' @keywords internal
register_study_backend_seed_methods <- function() {
  # Register with DelayedArray's generics (DelayedArray is in Imports)
  if (requireNamespace("S4Arrays", quietly = TRUE)) {
    # For newer versions that use S4Arrays (S4Arrays is in Suggests)
    registerS3method("extract_array", "study_backend_seed",
      extract_array.study_backend_seed,
      envir = asNamespace("S4Arrays")
    )
  } else {
    # For older versions
    registerS3method("extract_array", "study_backend_seed",
      extract_array.study_backend_seed,
      envir = asNamespace("DelayedArray")
    )
  }

  registerS3method("is_sparse", "study_backend_seed",
    is_sparse.study_backend_seed,
    envir = asNamespace("DelayedArray")
  )

  registerS3method("chunkGrid", "study_backend_seed",
    chunkGrid.study_backend_seed,
    envir = asNamespace("DelayedArray")
  )
}
