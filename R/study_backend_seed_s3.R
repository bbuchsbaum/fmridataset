#' Study Backend Seed for DelayedArray
#'
#' @description
#' A DelayedArray-compatible seed that provides lazy access to multi-subject fMRI data
#' without loading all subjects into memory at once. Implemented as an S4 class
#' inheriting from \code{Array} so it integrates natively with DelayedArray.
#'
#' @name study-backend-seed
#' @importFrom utils object.size
#' @keywords internal
NULL

.study_backend_seed_env <- new.env(parent = emptyenv())
.study_backend_seed_env$registered <- FALSE

#' Register Study Backend Seed Methods
#'
#' Ensures the DelayedArray-compatible seed and associated S4 methods are
#' available. Called automatically when study-level backends are coerced to
#' DelayedArray objects.
#'
#' @keywords internal
register_study_backend_seed_methods <- function() {
  if (isTRUE(.study_backend_seed_env$registered)) {
    return(invisible(NULL))
  }

  if (!.require_namespace("DelayedArray", quietly = TRUE)) {
    stop(
      "The DelayedArray package is required for DelayedArray study backends.",
      call. = FALSE
    )
  }

  methods::setClass(
    "StudyBackendSeed",
    slots = list(
      backends = "list",
      subject_ids = "character",
      subject_boundaries = "integer",
      dims = "integer",
      cache = "environment"
    ),
    contains = "Array"
  )

  extract_array_generic <- getExportedValue("DelayedArray", "extract_array")

  methods::setMethod("dim", "StudyBackendSeed", function(x) {
    x@dims
  })

  methods::setMethod("dimnames", "StudyBackendSeed", function(x) {
    list(NULL, NULL)
  })

  methods::setMethod(extract_array_generic, methods::signature(x = "StudyBackendSeed"), function(x, index) {
    if (!is.list(index) || length(index) != 2) {
      stop("index must be a list of length 2")
    }

    row_idx <- index[[1]]
    col_idx <- index[[2]]

    if (is.null(row_idx)) row_idx <- seq_len(x@dims[1])
    if (is.null(col_idx)) col_idx <- seq_len(x@dims[2])

    if (is.logical(row_idx)) row_idx <- which(row_idx)
    if (is.logical(col_idx)) col_idx <- which(col_idx)

    if (any(row_idx < 1L | row_idx > x@dims[1])) {
      stop("Row indices out of bounds")
    }
    if (any(col_idx < 1L | col_idx > x@dims[2])) {
      stop("Column indices out of bounds")
    }

    if (!length(row_idx) || !length(col_idx)) {
      return(matrix(numeric(), nrow = length(row_idx), ncol = length(col_idx)))
    }

    if (!is.integer(row_idx)) {
      if (is.double(row_idx) && all(row_idx == as.integer(row_idx))) {
        row_idx <- as.integer(row_idx)
      } else {
        stop("Row indices must be integer valued")
      }
    }

    if (!is.integer(col_idx)) {
      if (is.double(col_idx) && all(col_idx == as.integer(col_idx))) {
        col_idx <- as.integer(col_idx)
      } else {
        stop("Column indices must be integer valued")
      }
    }

    .collect_study_backend_block(
      backends = x@backends,
      rows = row_idx,
      cols = col_idx,
      subject_boundaries = x@subject_boundaries,
      n_time = x@dims[1],
      n_vox = x@dims[2]
    )
  })

  .study_backend_seed_env$registered <- TRUE
  invisible(NULL)
}

#' Create a Study Backend Seed
#'
#' @param backends List of subject backend objects
#' @param subject_ids Character vector of subject identifiers
#' @return A StudyBackendSeed S4 object
#' @keywords internal
study_backend_seed <- function(backends, subject_ids) {
  register_study_backend_seed_methods()

  if (!is.list(backends)) {
    stop("backends must be a list")
  }
  if (length(backends) != length(subject_ids)) {
    stop("Number of backends must match number of subject IDs")
  }

  # Get dimensions from each backend
  dims_list <- lapply(backends, backend_get_dims)
  dims_list <- lapply(dims_list, function(d) {
    d$spatial <- as.numeric(d$spatial)
    d$time <- as.integer(d$time)
    d
  })
  time_dims <- vapply(dims_list, function(x) x$time, integer(1))
  spatial_dims <- dims_list[[1]]$spatial
  n_voxels <- prod(spatial_dims)

  # Validate all backends have same spatial dimensions
  for (i in seq_along(dims_list)) {
    if (!identical(dims_list[[i]]$spatial, spatial_dims)) {
      stop(sprintf("Backend %d has inconsistent spatial dimensions", i))
    }
  }

  subject_boundaries <- c(0L, cumsum(time_dims))
  total_time <- sum(time_dims)
  dims <- c(total_time, n_voxels)

  cache <- create_study_cache()

  methods::new(
    "StudyBackendSeed",
    backends = backends,
    subject_ids = as.character(subject_ids),
    subject_boundaries = as.integer(subject_boundaries),
    dims = as.integer(dims),
    cache = cache
  )
}

#' Create LRU Cache for Study Backend
#' @keywords internal
create_study_cache <- function() {
  new.env(parent = emptyenv())
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
#' @keywords internal
study_seed_is_sparse <- function(x) {
  FALSE
}

#' Get Chunk Grid for Study Backend Seed
#'
#' @param x A study_backend_seed object
#' @param chunk_dim Optional chunk dimensions
#' @return A RegularArrayGrid object
#' @keywords internal
study_seed_chunk_grid <- function(x, chunk_dim = NULL) {
  if (!.require_namespace("DelayedArray", quietly = TRUE)) {
    stop("The DelayedArray package is required for chunk grid computation.", call. = FALSE)
  }

  if (is.null(chunk_dim)) {
    chunk_dim <- c(
      min(x@subject_boundaries[2] - x@subject_boundaries[1], x@dims[1]),
      x@dims[2]
    )
  }

  regular_grid <- getExportedValue("DelayedArray", "RegularArrayGrid")

  regular_grid(
    refdim = x@dims,
    spacings = chunk_dim
  )
}
