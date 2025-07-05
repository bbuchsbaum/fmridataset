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

  stopifnot(!isTRUE(getOption("DelayedArray.suppressWarnings")))
  options(fmridataset.block_size_mb = 64)

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

  backend <- list(
    backends = backends,
    subject_ids = subject_ids,
    strict = strict,
    `_dims` = list(spatial = ref_spatial, time = sum(time_dims)),
    `_mask` = combined_mask
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
  # Use the lazy DelayedArray approach
  da <- as_delayed_array(backend)
  
  # Subset if needed
  if (!is.null(rows) || !is.null(cols)) {
    # Convert NULL to full range
    if (is.null(rows)) rows <- seq_len(nrow(da))
    if (is.null(cols)) cols <- seq_len(ncol(da))
    
    # Extract only what's needed
    da[rows, cols, drop = FALSE]
  } else {
    da
  }
}
