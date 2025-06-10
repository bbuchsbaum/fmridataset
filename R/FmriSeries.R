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
