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

#' @rdname as_delarr
#' @export
as_delarr.study_backend <- function(backend, ...) {
  .ensure_delarr()

  if (is.null(backend$time_dims) || is.null(backend$subject_boundaries)) {
    dims_list <- lapply(backend$backends, backend_get_dims)
    backend$time_dims <- vapply(dims_list, function(d) as.integer(d$time), integer(1))
    backend$subject_boundaries <- c(0L, cumsum(backend$time_dims))
  }

  n_time <- sum(backend$time_dims)
  mask <- backend_get_mask(backend)
  n_vox <- as.integer(sum(mask))

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

    if (!is.integer(rows)) {
      if (is.double(rows) && all(rows == as.integer(rows))) {
        rows <- as.integer(rows)
      } else {
        stop("Row indices must be integer valued", call. = FALSE)
      }
    }

    if (!is.integer(cols)) {
      if (is.double(cols) && all(cols == as.integer(cols))) {
        cols <- as.integer(cols)
      } else {
        stop("Column indices must be integer valued", call. = FALSE)
      }
    }

    .collect_study_backend_block(
      backends = backend$backends,
      rows = rows,
      cols = cols,
      subject_boundaries = backend$subject_boundaries,
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
