#' fMRI Time Series Container
#'
#' An S4 class representing lazily accessed fMRI time series data. The
#' underlying data is stored in a `DelayedMatrix` with rows corresponding
#' to timepoints and columns corresponding to voxels.
#'
#' @slot voxel_info A `S4Vectors::DataFrame` containing spatial metadata
#'   for each voxel.
#' @slot temporal_info A `S4Vectors::DataFrame` containing metadata for
#'   each timepoint.
#' @slot selection_info A list describing how the data were selected.
#' @slot dataset_info A list describing the source dataset and backend.
#' @export
#' @importClassesFrom DelayedArray DelayedMatrix
#' @importClassesFrom S4Vectors DataFrame
setClass("FmriSeries",
         contains = "DelayedMatrix",
         slots = list(
           voxel_info = "DataFrame",
           temporal_info = "DataFrame",
           selection_info = "list",
           dataset_info = "list"
         ))

# Basic validity ----------------------------------------------------------

setValidity("FmriSeries", function(object) {
  msg <- NULL
  if (!inherits(object@voxel_info, "DataFrame")) {
    msg <- c(msg, "voxel_info must be a DataFrame")
  }
  if (!inherits(object@temporal_info, "DataFrame")) {
    msg <- c(msg, "temporal_info must be a DataFrame")
  }
  if (!is.list(object@selection_info)) {
    msg <- c(msg, "selection_info must be a list")
  }
  if (!is.list(object@dataset_info)) {
    msg <- c(msg, "dataset_info must be a list")
  }
  if (length(msg)) msg else TRUE
})

# Show method -------------------------------------------------------------

setMethod("show", "FmriSeries", function(object) {
  n_time <- nrow(object)
  n_vox <- ncol(object)
  cat(sprintf("<FmriSeries> %s voxels \u00d7 %s timepoints (lazy)\n",
              n_vox, n_time))
  sel <- object@selection_info
  dataset <- object@dataset_info
  sel_desc <- if (!is.null(sel$selector)) "custom" else "NULL"
  backend <- if (!is.null(dataset$backend_type)) dataset$backend_type else "?"
  cat(sprintf("Selector: %s | Backend: %s | Orientation: time \u00d7 voxels\n",
              sel_desc, backend))
})

# As.matrix and as_tibble methods
#' Materialise an FmriSeries as a standard matrix
#'
#' This method realises the underlying DelayedMatrix and
#' returns an ordinary matrix with timepoints in rows and
#' voxels in columns.
#'
#' @param x An `FmriSeries` object
#' @param ... Additional arguments (ignored)
#'
#' @return A matrix
#' @export
as.matrix.FmriSeries <- function(x, ...) {
  as.matrix(DelayedArray::DelayedArray(x))
}

#' Convert an FmriSeries to a tidy tibble
#'
#' The returned tibble contains one row per voxel/timepoint
#' combination with metadata columns from `temporal_info`
#' and `voxel_info` and a `signal` column with the data
#' values.
#'
#' @param x An `FmriSeries` object
#' @param ... Additional arguments (ignored)
#'
#' @return A tibble with signal and metadata columns
#' @export
#' @importFrom tibble as_tibble
as_tibble.FmriSeries <- function(x, ...) {
  mat <- as.matrix(x)
  vox_df <- as.data.frame(x@voxel_info)
  tmp_df <- as.data.frame(x@temporal_info)

  n_time <- nrow(mat)
  n_vox <- ncol(mat)
  time_idx <- rep(seq_len(n_time), times = n_vox)
  voxel_idx <- rep(seq_len(n_vox), each = n_time)

  out <- cbind(
    tmp_df[time_idx, , drop = FALSE],
    vox_df[voxel_idx, , drop = FALSE],
    signal = as.vector(mat)
  )
  tibble::as_tibble(out)
}
