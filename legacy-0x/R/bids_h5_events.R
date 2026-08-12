#' BIDS H5 Event and Confound Helpers
#'
#' @description
#' Internal functions for reading and writing events, confounds, and censor
#' vectors stored as column arrays inside a BIDS HDF5 archive.
#'
#' @details
#' Events are stored as one HDF5 dataset per column (column-array layout),
#' which gives better performance than compound datasets for variable-length
#' strings. The number of events is stored as the attribute \code{n_events}
#' on the parent group so readers can allocate correctly without probing lengths.
#'
#' Confounds are stored as a single \code{[T, n_confounds]} float64 matrix
#' dataset with a \code{names} attribute listing column names.
#'
#' Censor vectors are stored as \code{uint8} arrays of length T (0 = keep,
#' 1 = censor), matching the fmridataset convention.
#'
#' @name bids-h5-events
#' @keywords internal
NULL


# ============================================================
# Event writers
# ============================================================

#' Write Events to HDF5 Scan Group
#'
#' @description
#' Writes a data.frame (events table) to the \code{events/} subgroup of a
#' scan HDF5 group, using one dataset per column. Stores the \code{n_events}
#' attribute on the group.
#'
#' @param h5_group An \code{hdf5r::H5Group} object for the scan (e.g.
#'   \code{h5file[["scans/sub-01_task-nback_run-01"]]}).
#' @param events A data.frame with event columns (onset, duration, trial_type,
#'   ...). Must have at least \code{onset} and \code{duration}.
#' @param compression Integer 0-9. HDF5 gzip compression level (default 4).
#'
#' @return Invisible NULL.
#' @keywords internal
h5_write_events <- function(h5_group, events, compression = 4L) {
  if (is.null(events) || nrow(events) == 0L) {
    return(invisible(NULL))
  }

  if (!h5_group$exists("events")) {
    h5_group$create_group("events")
  }
  ev_grp <- h5_group[["events"]]

  n_events <- nrow(events)
  # Store n_events as a group attribute for fast access
  ev_grp$create_attr("n_events", n_events)

  for (col_name in names(events)) {
    col_data <- events[[col_name]]

    if (is.factor(col_data)) {
      col_data <- as.character(col_data)
    } else if (!is.character(col_data)) {
      col_data <- as.double(col_data)
    }

    # Use robj= so hdf5r infers the correct type (including variable-length
    # strings for character vectors). Explicit dtype+space+write() requires
    # the hdf5r subset-assignment syntax and is more fragile.
    ev_grp$create_dataset(
      col_name,
      robj       = col_data,
      chunk_dims = min(n_events, 1000L),
      gzip_level = compression
    )
  }

  invisible(NULL)
}


# ============================================================
# Event readers
# ============================================================

#' Read Events from HDF5 Scan Group
#'
#' @description
#' Reads the column-array events stored under \code{events/} in a scan group
#' and reassembles them into a data.frame. Returns \code{NULL} if the group
#' has no \code{events/} subgroup or the group is empty.
#'
#' @param h5_group An \code{hdf5r::H5Group} for the scan.
#'
#' @return A data.frame of events, or \code{NULL}.
#' @keywords internal
h5_read_events <- function(h5_group) {
  if (!h5_group$exists("events")) {
    return(NULL)
  }

  ev_grp <- h5_group[["events"]]
  col_names <- ev_grp$ls()$name

  if (length(col_names) == 0L) {
    return(NULL)
  }

  cols <- lapply(col_names, function(cn) {
    ev_grp[[cn]]$read()
  })
  names(cols) <- col_names

  as.data.frame(cols, stringsAsFactors = FALSE)
}


# ============================================================
# Confound writers / readers
# ============================================================

#' Write Confound Matrix to HDF5 Scan Group
#'
#' @description
#' Writes a confound regressor matrix to \code{confounds/data} in the scan
#' group, storing column names as a \code{names} attribute.
#'
#' @param h5_group An \code{hdf5r::H5Group} for the scan.
#' @param confounds A matrix or data.frame \code{[T, n_confounds]}. If
#'   \code{NULL}, nothing is written.
#' @param compression Integer 0-9. HDF5 gzip compression level (default 4).
#'
#' @return Invisible NULL.
#' @keywords internal
h5_write_confounds <- function(h5_group, confounds, compression = 4L) {
  if (is.null(confounds)) {
    return(invisible(NULL))
  }

  col_names <- if (is.data.frame(confounds)) names(confounds) else colnames(confounds)
  mat <- as.matrix(confounds)
  storage.mode(mat) <- "double"

  if (!h5_group$exists("confounds")) {
    h5_group$create_group("confounds")
  }
  cf_grp <- h5_group[["confounds"]]

  n_time     <- nrow(mat)
  n_confounds <- ncol(mat)
  chunks <- c(min(n_time, 256L), min(n_confounds, 64L))

  ds <- cf_grp$create_dataset(
    "data",
    robj       = mat,
    chunk_dims = chunks,
    gzip_level = compression
  )

  if (!is.null(col_names)) {
    ds$create_attr("names", col_names)
  }

  invisible(NULL)
}

#' Read Confound Matrix from HDF5 Scan Group
#'
#' @description
#' Reads the confound matrix stored under \code{confounds/data} and returns
#' a data.frame with the original column names restored from the \code{names}
#' attribute. Returns \code{NULL} if no confounds are stored.
#'
#' @param h5_group An \code{hdf5r::H5Group} for the scan.
#'
#' @return A data.frame \code{[T, n_confounds]}, or \code{NULL}.
#' @keywords internal
h5_read_confounds <- function(h5_group) {
  if (!h5_group$exists("confounds")) {
    return(NULL)
  }
  cf_grp <- h5_group[["confounds"]]
  if (!cf_grp$exists("data")) {
    return(NULL)
  }

  ds  <- cf_grp[["data"]]
  mat <- ds$read()

  # Restore column names from attribute if present
  if (ds$attr_exists("names")) {
    col_names <- ds$attr_open("names")$read()
  } else {
    col_names <- paste0("confound_", seq_len(ncol(mat)))
  }

  df <- as.data.frame(mat, stringsAsFactors = FALSE)
  names(df) <- col_names
  df
}


# ============================================================
# Censor vector writers / readers
# ============================================================

#' Write Censor Vector to HDF5 Scan Group
#'
#' @description
#' Writes a logical or integer censor vector as a \code{uint8} dataset named
#' \code{censor} in the scan group. Values: 0 = keep, 1 = censor.
#'
#' @param h5_group An \code{hdf5r::H5Group} for the scan.
#' @param censor Integer or logical vector of length T. \code{NULL} means no
#'   censor vector is written (all timepoints kept).
#' @param compression Integer 0-9 (default 4).
#'
#' @return Invisible NULL.
#' @keywords internal
h5_write_censor <- function(h5_group, censor, compression = 4L) {
  if (is.null(censor)) {
    return(invisible(NULL))
  }

  # Store as integer (0/1); hdf5r will use a native int type via robj=
  censor_int <- as.integer(as.logical(censor))
  n_time <- length(censor_int)

  h5_group$create_dataset(
    "censor",
    robj       = censor_int,
    chunk_dims = min(n_time, 1024L),
    gzip_level = compression
  )

  invisible(NULL)
}

#' Read Censor Vector from HDF5 Scan Group
#'
#' @description
#' Reads the \code{censor} dataset from a scan group. Returns \code{NULL} if
#' absent (implying all timepoints are kept).
#'
#' @param h5_group An \code{hdf5r::H5Group} for the scan.
#'
#' @return Integer vector (0/1) of length T, or \code{NULL}.
#' @keywords internal
h5_read_censor <- function(h5_group) {
  if (!h5_group$exists("censor")) {
    return(NULL)
  }
  as.integer(h5_group[["censor"]]$read())
}


# ============================================================
# Scan metadata writers / readers
# ============================================================

#' Write Per-Scan Metadata to HDF5
#'
#' @description
#' Writes scalar metadata fields (subject, task, session, run, tr) as
#' individual string/numeric datasets under \code{metadata/} in a scan group.
#'
#' @param h5_group An \code{hdf5r::H5Group} for the scan.
#' @param meta Named list with scalar elements: subject, task, session (may be
#'   NULL), run, tr.
#'
#' @return Invisible NULL.
#' @keywords internal
h5_write_scan_metadata <- function(h5_group, meta) {
  if (!h5_group$exists("metadata")) {
    h5_group$create_group("metadata")
  }
  md_grp <- h5_group[["metadata"]]

  for (key in names(meta)) {
    val <- meta[[key]]
    if (is.null(val)) next
    md_grp$create_dataset(key, robj = val)
  }

  invisible(NULL)
}

#' Read Per-Scan Metadata from HDF5
#'
#' @description
#' Reads the scalar metadata datasets from \code{metadata/} in a scan group.
#'
#' @param h5_group An \code{hdf5r::H5Group} for the scan.
#'
#' @return Named list of metadata values.
#' @keywords internal
h5_read_scan_metadata <- function(h5_group) {
  if (!h5_group$exists("metadata")) {
    return(list())
  }
  md_grp <- h5_group[["metadata"]]
  keys <- md_grp$ls()$name

  meta <- lapply(keys, function(k) md_grp[[k]]$read())
  names(meta) <- keys
  meta
}
