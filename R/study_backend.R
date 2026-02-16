#' Study Backend
#'
#' Composite backend that lazily combines multiple subject-level backends.
#'
#' @param backends list of storage_backend objects
#' @param subject_ids vector of subject identifiers matching `backends`
#' @param strict mask validation mode. "identical" or "intersect"
#' @return A `study_backend` object
#' @export
study_backend <- function(backends, subject_ids = NULL,
                          strict = getOption("fmridataset.mask_check", "identical")) {
  if (!is.list(backends) || length(backends) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "backends must be a non-empty list"
    )
  }

  # Coerce fmri_dataset objects to their backends
  backends <- lapply(backends, function(b) {
    if (!inherits(b, "storage_backend")) {
      if (inherits(b, "matrix_dataset") && !is.null(b$datamat)) {
        # Legacy matrix_dataset - convert to matrix_backend
        mask_logical <- as.logical(b$mask)
        matrix_backend(b$datamat, mask = mask_logical)
      } else if (!is.null(b$backend)) {
        # New-style dataset with backend
        b$backend
      } else {
        # Return as-is and let validation catch it
        b
      }
    } else {
      b
    }
  })

  lapply(backends, function(b) {
    if (!inherits(b, "storage_backend")) {
      stop_fmridataset(
        fmridataset_error_config,
        message = "all elements of backends must inherit from 'storage_backend'"
      )
    }
  })

  if (is.null(subject_ids)) {
    subject_ids <- seq_along(backends)
  }

  if (length(subject_ids) != length(backends)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "subject_ids must match length of backends"
    )
  }

  dims_list <- lapply(backends, backend_get_dims)
  # Ensure consistent numeric type for spatial dimensions
  spatial_dims <- lapply(dims_list, function(x) as.numeric(x$spatial))
  time_dims <- vapply(dims_list, function(x) x$time, numeric(1))

  ref_spatial <- spatial_dims[[1]]
  for (i in seq_along(spatial_dims[-1])) {
    sd <- spatial_dims[[i + 1]]
    if (!identical(sd, ref_spatial)) {
      stop_fmridataset(
        fmridataset_error_config,
        message = "spatial dimensions must match across backends"
      )
    }
  }

  masks <- lapply(backends, backend_get_mask)
  ref_mask <- masks[[1]]
  if (strict == "identical") {
    for (m in masks[-1]) {
      if (!identical(m, ref_mask)) {
        stop_fmridataset(
          fmridataset_error_config,
          message = "masks differ across backends"
        )
      }
    }
    combined_mask <- ref_mask
  } else if (strict == "intersect") {
    for (m in masks[-1]) {
      overlap <- sum(m & ref_mask) / length(ref_mask)
      if (overlap < 0.95) {
        stop_fmridataset(
          fmridataset_error_config,
          message = "mask overlap <95%"
        )
      }
    }
    combined_mask <- Reduce("&", masks)
  } else {
    stop_fmridataset(
      fmridataset_error_config,
      message = "unknown strict setting"
    )
  }

  subject_boundaries <- c(0L, cumsum(as.integer(time_dims)))

  backend <- list(
    backends = backends,
    subject_ids = subject_ids,
    strict = strict,
    `_dims` = list(spatial = ref_spatial, time = sum(time_dims)),
    `_mask` = combined_mask,
    time_dims = as.integer(time_dims),
    subject_boundaries = as.integer(subject_boundaries)
  )
  class(backend) <- c("study_backend", "storage_backend")
  backend
}

#' @rdname backend_open
#' @method backend_open study_backend
#' @export
backend_open.study_backend <- function(backend) {
  backend$backends <- lapply(backend$backends, backend_open)
  backend
}

#' @rdname backend_close
#' @method backend_close study_backend
#' @export
backend_close.study_backend <- function(backend) {
  lapply(backend$backends, backend_close)
  invisible(NULL)
}

#' @rdname backend_get_dims
#' @method backend_get_dims study_backend
#' @export
backend_get_dims.study_backend <- function(backend) {
  backend$`_dims`
}

#' @rdname backend_get_mask
#' @method backend_get_mask study_backend
#' @export
backend_get_mask.study_backend <- function(backend) {
  backend$`_mask`
}

#' @rdname backend_get_data
#' @method backend_get_data study_backend
#' @export
backend_get_data.study_backend <- function(backend, rows = NULL, cols = NULL) {
  if (is.null(backend$time_dims) || is.null(backend$subject_boundaries)) {
    dims_list <- lapply(backend$backends, backend_get_dims)
    backend$time_dims <- vapply(dims_list, function(d) as.integer(d$time), integer(1))
    backend$subject_boundaries <- c(0L, cumsum(backend$time_dims))
  }

  n_time <- sum(backend$time_dims)
  mask <- backend_get_mask(backend)
  n_vox <- as.integer(sum(mask))

  if (is.null(rows) && is.null(cols) && requireNamespace("delarr", quietly = TRUE)) {
    return(as_delarr(backend))
  }

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

#' @rdname backend_get_metadata
#' @method backend_get_metadata study_backend
#' @export
backend_get_metadata.study_backend <- function(backend) {
  # Collect metadata from first subject backend as representative
  first_meta <- tryCatch(
    backend_get_metadata(backend$backends[[1]]),
    error = function(e) list()
  )
  first_meta$storage_format <- "study"
  first_meta$n_subjects <- length(backend$backends)
  first_meta$subject_ids <- backend$subject_ids
  first_meta
}

.collect_study_backend_block <- function(backends, rows, cols,
                                         subject_boundaries, n_time, n_vox) {
  n_rows <- length(rows)
  n_cols <- length(cols)

  if (!n_rows || !n_cols) {
    return(matrix(numeric(), nrow = n_rows, ncol = n_cols))
  }

  ord <- order(rows)
  sorted_rows <- rows[ord]
  result_sorted <- matrix(NA_real_, nrow = n_rows, ncol = n_cols)

  for (s in seq_along(backends)) {
    start <- subject_boundaries[s] + 1L
    end <- subject_boundaries[s + 1L]
    idx <- which(sorted_rows >= start & sorted_rows <= end)
    if (!length(idx)) next

    subj_backend <- backends[[s]]
    local_rows <- sorted_rows[idx] - subject_boundaries[s]
    subj_data <- backend_get_data(subj_backend, rows = local_rows, cols = cols)
    if (!is.matrix(subj_data)) {
      subj_data <- as.matrix(subj_data)
    }

    result_sorted[idx, ] <- subj_data
  }

  result <- matrix(NA_real_, nrow = n_rows, ncol = n_cols)
  result[ord, ] <- result_sorted
  result
}
