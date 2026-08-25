#' Convert backend to a delarr lazy matrix
#'
#' Provides a lightweight S3 interface that defers materialization of backend
#' data. The returned object is compatible with `delarr::collect()` as well as
#' base `as.matrix()` for realization.
#'
#' @param backend A storage backend object
#' @param ... Passed to methods
#' @return A `delarr` lazy matrix
#' @export
as_delarr <- function(backend, ...) {
  UseMethod("as_delarr")
}

.ensure_delarr <- function() {
  if (!requireNamespace("delarr", quietly = TRUE)) {
    stop(
      "The delarr package is required for lazy matrix operations.",
      call. = FALSE
    )
  }
}

#' @rdname as_delarr
#' @export
as_delarr.matrix_backend <- function(backend, ...) {
  .ensure_delarr()
  dims <- backend_get_dims(backend)
  mask <- backend_get_mask(backend)
  n_time <- as.integer(dims$time)
  n_vox <- as.integer(sum(mask))

  delarr::delarr_backend(
    nrow = n_time,
    ncol = n_vox,
    pull = function(rows = NULL, cols = NULL) {
      backend_get_data(backend, rows = rows, cols = cols)
    }
  )
}

#' @rdname as_delarr
#' @export
as_delarr.nifti_backend <- function(backend, ...) {
  .ensure_delarr()
  dims <- backend_get_dims(backend)
  mask <- backend_get_mask(backend)
  n_time <- as.integer(dims$time)
  n_vox <- as.integer(sum(mask))

  delarr::delarr_backend(
    nrow = n_time,
    ncol = n_vox,
    pull = function(rows = NULL, cols = NULL) {
      backend_get_data(backend, rows = rows, cols = cols)
    }
  )
}

# Ensure the study backend has cached per-subject time dims and boundaries.
.as_delarr_study_ensure_dims <- function(backend) {
  if (is.null(backend$time_dims) || is.null(backend$subject_boundaries)) {
    dims_list <- lapply(backend$backends, backend_get_dims)
    backend$time_dims <- vapply(dims_list, function(d) as.integer(d$time), integer(1))
    backend$subject_boundaries <- c(0L, cumsum(backend$time_dims))
  }
  backend
}

# Coerce a single index vector (rows or cols) to validated integer indices.
# `label` selects the error wording ("Row" or "Column").
.as_delarr_study_coerce_index <- function(idx, n, label) {
  if (is.logical(idx)) idx <- which(idx)

  if (any(idx < 1L | idx > n)) {
    stop(label, " indices out of bounds", call. = FALSE)
  }

  if (!is.integer(idx)) {
    if (is.double(idx) && all(idx == as.integer(idx))) {
      idx <- as.integer(idx)
    } else {
      stop(label, " indices must be integer valued", call. = FALSE)
    }
  }

  idx
}

#' @rdname as_delarr
#' @export
as_delarr.study_backend <- function(backend, ...) {
  .ensure_delarr()

  backend <- .as_delarr_study_ensure_dims(backend)

  n_time <- sum(backend$time_dims)
  mask <- backend_get_mask(backend)
  n_vox <- as.integer(sum(mask))
  # Resolved once, outside the closure: this is the default read path for a
  # study backend, and it hands child backends columns numbered in the
  # combined mask.
  col_maps <- .study_backend_resolve_col_maps(backend, mask)

  pull_fun <- function(rows = NULL, cols = NULL) {
    rows <- if (is.null(rows)) seq_len(n_time) else rows
    cols <- if (is.null(cols)) seq_len(n_vox) else cols

    if (is.logical(rows)) rows <- which(rows)
    if (is.logical(cols)) cols <- which(cols)

    if (any(rows < 1L | rows > n_time)) {
      stop("Row indices out of bounds", call. = FALSE)
    }
    if (any(cols < 1L | cols > n_vox)) {
      stop("Column indices out of bounds", call. = FALSE)
    }

    if (!length(rows) || !length(cols)) {
      return(matrix(numeric(), nrow = length(rows), ncol = length(cols)))
    }

    rows <- .as_delarr_study_coerce_index(rows, n_time, "Row")
    cols <- .as_delarr_study_coerce_index(cols, n_vox, "Column")

    .collect_study_backend_block(
      backends = backend$backends,
      rows = rows,
      cols = cols,
      subject_boundaries = backend$subject_boundaries,
      col_maps = col_maps,
      n_time = n_time,
      n_vox = n_vox
    )
  }

  delarr::delarr_backend(
    nrow = n_time,
    ncol = n_vox,
    pull = pull_fun
  )
}

#' @rdname as_delarr
#' @export
as_delarr.default <- function(backend, ...) {
  stop("No as_delarr method for class: ", class(backend)[1])
}
