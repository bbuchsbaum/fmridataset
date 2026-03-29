#' BIDS H5 Scan Backend
#'
#' @description
#' A lightweight storage backend for a single scan stored inside a shared
#' BIDS HDF5 archive. Many \code{bids_h5_scan_backend} objects share a single
#' \code{h5_shared_connection}, allowing one file handle to serve an entire
#' study without leaking file descriptors.
#'
#' @details
#' The backend operates in **feature-space**: columns are K features
#' (parcels for \code{compression_mode = "parcellated"}, or latent components
#' for \code{compression_mode = "latent"}), not V voxels. This satisfies the
#' backend contract by reporting \code{spatial = c(K, 1, 1)} and
#' \code{mask = rep(TRUE, K)}, so \code{validate_backend()} passes and all
#' downstream components (study_backend, as_delarr, data_chunks) work unchanged.
#'
#' For \code{"parcellated"} mode, data is read from
#' \code{/scans/<name>/data/summary_data} (shape \code{[T, K]}).
#' For \code{"latent"} mode, the temporal basis is read from
#' \code{/scans/<name>/data/basis} (shape \code{[T, K]}). Loadings and offset
#' can be accessed via \code{.read_scan_loadings()} and
#' \code{.read_scan_offset()} helpers.
#'
#' Original voxel geometry is stored in the HDF5 file under \code{/spatial/}
#' and (for parcellated mode) \code{/parcellation/}, but does **not** flow
#' through the backend contract.
#'
#' @name bids-h5-backend
#' @keywords internal
NULL


# ============================================================
# Shared H5 Connection (ref-counted)
# ============================================================

#' Create a Shared H5 Connection
#'
#' @description
#' A ref-counted wrapper around an \code{hdf5r::H5File} object. Multiple
#' \code{bids_h5_scan_backend} objects share one connection; the file is closed
#' only when the last backend releases it.
#'
#' @param file Character string. Path to the HDF5 file to open.
#'
#' @return An environment of class \code{h5_shared_connection} with fields:
#'   \itemize{
#'     \item \code{file}: the file path
#'     \item \code{handle}: the open \code{hdf5r::H5File} object
#'     \item \code{ref_count}: integer, number of live backends holding this connection
#'   }
#'   And methods \code{acquire()} and \code{release()}.
#'
#' @keywords internal
#' @export
h5_shared_connection <- function(file) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "Package 'hdf5r' is required for BIDS H5 backend but is not available",
      parameter = "file"
    )
  }

  if (!file.exists(file)) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf("BIDS H5 file not found: %s", file),
      file = file,
      operation = "open"
    )
  }

  conn <- new.env(parent = emptyenv())
  conn$file <- file
  conn$ref_count <- 0L
  conn$handle <- NULL

  conn$open_handle <- function() {
    conn$handle <- tryCatch(
      hdf5r::H5File$new(file, mode = "r"),
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          message = sprintf("Failed to open BIDS H5 file '%s': %s", file, e$message),
          file = file,
          operation = "open"
        )
      }
    )
    invisible(NULL)
  }

  # Open file immediately — shared handle lives for the connection lifetime
  conn$open_handle()

  conn$acquire <- function() {
    if (is.null(conn$handle) || !isTRUE(conn$handle$is_valid)) {
      conn$open_handle()
    }
    conn$ref_count <- conn$ref_count + 1L
    invisible(NULL)
  }

  conn$release <- function() {
    conn$ref_count <- conn$ref_count - 1L
    if (conn$ref_count <= 0L) {
      if (!is.null(conn$handle) && isTRUE(conn$handle$is_valid)) {
        tryCatch(conn$handle$close_all(), error = function(e) invisible(NULL))
      }
      conn$ref_count <- 0L
    }
    invisible(NULL)
  }

  class(conn) <- "h5_shared_connection"
  conn
}

#' @export
print.h5_shared_connection <- function(x, ...) {
  cat(sprintf(
    "<h5_shared_connection> file=%s ref_count=%d is_valid=%s\n",
    x$file, x$ref_count,
    if (x$handle$is_valid) "TRUE" else "FALSE"
  ))
  invisible(x)
}


# ============================================================
# bids_h5_scan_backend constructor
# ============================================================

#' Create a BIDS H5 Scan Backend
#'
#' @description
#' Constructs a lightweight backend for one scan stored in a BIDS HDF5 archive.
#' The backend holds a reference to a shared \code{h5_shared_connection} and
#' reads data from the HDF5 group for this scan. For \code{"parcellated"} mode
#' data is at \code{/scans/<name>/data/summary_data}; for \code{"latent"} mode
#' the temporal basis is at \code{/scans/<name>/data/basis}.
#'
#' @param h5_connection An \code{h5_shared_connection} object (shared across scans).
#' @param scan_group_path Character string. HDF5 group path for this scan,
#'   e.g. \code{"/scans/sub-01_task-nback_run-01"}.
#' @param n_features Integer. Number of features (parcels or latent components)
#'   — i.e. the number of columns in the data matrix.
#' @param n_time Integer. Number of timepoints (rows in the data matrix).
#' @param metadata Named list of scan metadata (subject, task, session, run, tr).
#'   May be empty; defaults to \code{list()}.
#' @param compression_mode Character. Either \code{"parcellated"} (default) or
#'   \code{"latent"}. Determines which HDF5 dataset is read by
#'   \code{backend_get_data}.
#'
#' @return A \code{bids_h5_scan_backend} / \code{storage_backend} environment.
#'
#' @keywords internal
#' @export
bids_h5_scan_backend <- function(h5_connection,
                                  scan_group_path,
                                  n_features,
                                  n_time,
                                  metadata = list(),
                                  compression_mode = "parcellated") {
  if (!inherits(h5_connection, "h5_shared_connection")) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "h5_connection must be an h5_shared_connection object",
      parameter = "h5_connection"
    )
  }

  compression_mode <- match.arg(compression_mode, c("parcellated", "latent"))
  n_features <- as.integer(n_features)
  n_time     <- as.integer(n_time)

  if (n_features < 1L) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "n_features must be a positive integer",
      parameter = "n_features",
      value = n_features
    )
  }
  if (n_time < 1L) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "n_time must be a positive integer",
      parameter = "n_time",
      value = n_time
    )
  }

  backend <- new.env(parent = emptyenv())
  backend$h5_connection    <- h5_connection
  backend$scan_group_path  <- scan_group_path
  backend$n_features       <- n_features
  backend$n_time           <- n_time
  backend$metadata         <- metadata
  backend$compression_mode <- compression_mode
  backend$is_open          <- FALSE

  class(backend) <- c("bids_h5_scan_backend", "storage_backend")
  backend
}


# ============================================================
# Backend contract methods
# ============================================================

#' @rdname backend_open
#' @method backend_open bids_h5_scan_backend
#' @export
backend_open.bids_h5_scan_backend <- function(backend) {
  if (!backend$is_open) {
    backend$h5_connection$acquire()
    backend$is_open <- TRUE
  }
  backend
}

#' @rdname backend_close
#' @method backend_close bids_h5_scan_backend
#' @export
backend_close.bids_h5_scan_backend <- function(backend) {
  if (backend$is_open) {
    backend$h5_connection$release()
    backend$is_open <- FALSE
  }
  invisible(NULL)
}

#' @rdname backend_get_dims
#' @method backend_get_dims bids_h5_scan_backend
#' @export
backend_get_dims.bids_h5_scan_backend <- function(backend) {
  # Feature-space: treat K features (parcels or latent components) as a
  # pseudo-3D volume with dimensions (K, 1, 1) so validate_backend() and
  # study_backend pass unchanged.
  list(
    spatial = c(backend$n_features, 1L, 1L),
    time    = backend$n_time
  )
}

#' @rdname backend_get_mask
#' @method backend_get_mask bids_h5_scan_backend
#' @export
backend_get_mask.bids_h5_scan_backend <- function(backend) {
  # All K feature columns are valid — all TRUE mask of length K.
  rep(TRUE, backend$n_features)
}

#' @rdname backend_get_data
#' @method backend_get_data bids_h5_scan_backend
#' @export
#' @details
#' For \code{compression_mode = "parcellated"} reads
#' \code{/scans/<name>/data/summary_data} (\code{[T, K]}).
#' For \code{compression_mode = "latent"} reads
#' \code{/scans/<name>/data/basis} (\code{[T, K]}).
#' In both cases the return value is a \code{[T, K]} numeric matrix.
backend_get_data.bids_h5_scan_backend <- function(backend, rows = NULL, cols = NULL) {
  conn <- backend$h5_connection

  if (!conn$handle$is_valid) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf(
        "H5 file handle for '%s' is no longer valid (file may have been closed)",
        conn$file
      ),
      file = conn$file,
      operation = "read"
    )
  }

  dataset_path <- if (backend$compression_mode == "latent") {
    paste0(backend$scan_group_path, "/data/basis")
  } else {
    paste0(backend$scan_group_path, "/data/summary_data")
  }

  data_matrix <- tryCatch({
    ds <- conn$handle[[dataset_path]]
    # HDF5 dataset is stored [T, K]; read as matrix and ensure correct orientation
    mat <- ds[, ]   # shape [T, K] from hdf5r (row-major read)
    if (!is.matrix(mat)) {
      mat <- matrix(mat, nrow = backend$n_time, ncol = backend$n_features)
    }
    mat
  }, error = function(e) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf(
        "Failed to read data from '%s': %s", dataset_path, e$message
      ),
      file = conn$file,
      operation = "read"
    )
  })

  # Apply row/col subsetting
  if (!is.null(rows)) {
    data_matrix <- data_matrix[rows, , drop = FALSE]
  }
  if (!is.null(cols)) {
    data_matrix <- data_matrix[, cols, drop = FALSE]
  }

  data_matrix
}

#' @rdname backend_get_metadata
#' @method backend_get_metadata bids_h5_scan_backend
#' @export
backend_get_metadata.bids_h5_scan_backend <- function(backend) {
  meta <- backend$metadata
  meta$compression_mode <- backend$compression_mode
  meta$n_features       <- backend$n_features
  meta$scan_group_path  <- backend$scan_group_path
  meta$format           <- "bids_h5"
  meta
}


# ============================================================
# Print method
# ============================================================

#' @export
print.bids_h5_scan_backend <- function(x, ...) {
  cat(sprintf(
    "<bids_h5_scan_backend>\n  scan: %s\n  mode: %s  features: %d  timepoints: %d\n  open: %s\n",
    x$scan_group_path, x$compression_mode, x$n_features, x$n_time,
    if (x$is_open) "yes" else "no"
  ))
  invisible(x)
}


# ============================================================
# Latent-mode helpers
# ============================================================

#' Read spatial loadings for a latent-mode scan
#'
#' @description
#' Reads the \code{[V, K]} loadings matrix stored at
#' \code{<scan_group_path>/data/loadings} in a BIDS HDF5 archive.
#' Only meaningful when \code{compression_mode = "latent"}.
#'
#' @param h5_handle An open \code{hdf5r::H5File} object.
#' @param scan_group_path Character string. HDF5 group path for the scan.
#'
#' @return A numeric matrix of shape \code{[V, K]}, or \code{NULL} if the
#'   dataset is absent (e.g. shared template mode where per-scan loadings are
#'   not stored).
#' @keywords internal
.read_scan_loadings <- function(h5_handle, scan_group_path) {
  path <- paste0(scan_group_path, "/data/loadings")
  if (!h5_handle$exists(path)) return(NULL)
  h5_handle[[path]][, ]
}

#' Check if archive has a shared template
#'
#' @param h5_handle An open \code{hdf5r::H5File} object.
#' @return Logical.
#' @keywords internal
.has_shared_template <- function(h5_handle) {
  h5_handle$exists("latent_meta/has_shared_template") &&
    isTRUE(h5_handle[["latent_meta/has_shared_template"]]$read())
}

#' Read shared template loadings
#'
#' @param h5_handle An open \code{hdf5r::H5File} object.
#' @return A numeric matrix \code{[V, K]} or \code{NULL}.
#' @keywords internal
.read_template_loadings <- function(h5_handle) {
  path <- "latent_meta/template/loadings"
  if (!h5_handle$exists(path)) return(NULL)
  h5_handle[[path]][, ]
}

#' Read per-voxel offset for a latent-mode scan
#'
#' @description
#' Reads the \code{[V]} offset vector stored at
#' \code{<scan_group_path>/data/offset} in a BIDS HDF5 archive.
#' Returns \code{numeric(0)} if the dataset is absent (offset not stored).
#' Only meaningful when \code{compression_mode = "latent"}.
#'
#' @param h5_handle An open \code{hdf5r::H5File} object.
#' @param scan_group_path Character string. HDF5 group path for the scan.
#'
#' @return A numeric vector of length V, or \code{numeric(0)}.
#' @keywords internal
.read_scan_offset <- function(h5_handle, scan_group_path) {
  path <- paste0(scan_group_path, "/data/offset")
  if (h5_handle$exists(path)) h5_handle[[path]][] else numeric(0)
}
