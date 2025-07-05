#' FmriSeries: fMRI Time Series Container
#'
#' @description
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
#' 
#' @return An object of class \code{FmriSeries} inheriting from \code{DelayedMatrix}
#' 
#' @seealso 
#' \code{\link{as.matrix.FmriSeries}} for converting to standard matrix,
#' \code{\link{as_tibble.FmriSeries}} for converting to tibble format
#' 
#' @examples
#' \donttest{
#' # Create example FmriSeries object
#' if (requireNamespace("DelayedArray", quietly = TRUE) && 
#'     requireNamespace("S4Vectors", quietly = TRUE)) {
#'   # Create small example data
#'   mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
#'   delayed_mat <- DelayedArray::DelayedArray(mat)
#'   
#'   # Create metadata
#'   vox_info <- S4Vectors::DataFrame(
#'     x = rep(1:10, 5),
#'     y = rep(1:5, each = 10),
#'     z = 1
#'   )
#'   
#'   temp_info <- S4Vectors::DataFrame(
#'     time = seq(0, 99, by = 1),
#'     run = rep(1:4, each = 25)
#'   )
#'   
#'   # Note: Direct instantiation typically not recommended
#'   # Users should use constructor functions provided by the package
#' }
#' }
#' 
#' @export
#' @importClassesFrom DelayedArray DelayedMatrix
#' @importClassesFrom S4Vectors DataFrame
#' @importFrom methods setClass setMethod setValidity new
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

#' Show Method for FmriSeries Objects
#'
#' @description
#' Display a concise summary of an FmriSeries object, including dimensions,
#' selector type, backend, and data orientation.
#'
#' @param object An \code{FmriSeries} object
#' 
#' @return Returns \code{object} invisibly. Called for its side effect of
#'   printing to the console.
#' 
#' @examples
#' \donttest{
#' # This method is called automatically when printing
#' if (requireNamespace("DelayedArray", quietly = TRUE) && 
#'     requireNamespace("S4Vectors", quietly = TRUE)) {
#'   # Create example object (see FmriSeries class documentation)
#'   # fs <- new("FmriSeries", ...)
#'   # fs  # Automatically calls show method
#' }
#' }
#' 
#' @rdname FmriSeries-class
#' @importFrom methods show
#' @exportMethod show
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
  invisible(object)
})

# As.matrix and as_tibble methods
#' Convert FmriSeries to Matrix
#'
#' @description
#' This method realises the underlying DelayedMatrix and
#' returns an ordinary matrix with timepoints in rows and
#' voxels in columns.
#'
#' @param x An \code{FmriSeries} object
#' @param ... Additional arguments (ignored)
#'
#' @return A matrix with timepoints as rows and voxels as columns
#' 
#' @seealso 
#' \code{\link{FmriSeries-class}} for the class definition,
#' \code{\link{as_tibble.FmriSeries}} for tibble conversion
#' 
#' @examples
#' \donttest{
#' if (requireNamespace("DelayedArray", quietly = TRUE) && 
#'     requireNamespace("S4Vectors", quietly = TRUE)) {
#'   # Create small example
#'   mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
#'   delayed_mat <- DelayedArray::DelayedArray(mat)
#'   
#'   # Create minimal FmriSeries object
#'   fs <- new("FmriSeries", delayed_mat,
#'             voxel_info = S4Vectors::DataFrame(idx = 1:5),
#'             temporal_info = S4Vectors::DataFrame(time = 1:4),
#'             selection_info = list(),
#'             dataset_info = list())
#'   
#'   # Convert to matrix
#'   mat_result <- as.matrix(fs)
#'   dim(mat_result)  # 4 x 5
#' }
#' }
#' 
#' @export
as.matrix.FmriSeries <- function(x, ...) {
  # FmriSeries already inherits from DelayedMatrix, so we can use its as.matrix method
  DelayedArray::as.matrix(x)
}

#' Convert FmriSeries to Tibble
#'
#' @description
#' The returned tibble contains one row per voxel/timepoint
#' combination with metadata columns from \code{temporal_info}
#' and \code{voxel_info} and a \code{signal} column with the data
#' values.
#'
#' @param x An \code{FmriSeries} object
#' @param ... Additional arguments (ignored)
#'
#' @return A tibble with columns from temporal_info, voxel_info, and a 
#'   signal column containing the fMRI signal values
#' 
#' @seealso 
#' \code{\link{FmriSeries-class}} for the class definition,
#' \code{\link{as.matrix.FmriSeries}} for matrix conversion
#' 
#' @examples
#' \donttest{
#' if (requireNamespace("DelayedArray", quietly = TRUE) && 
#'     requireNamespace("S4Vectors", quietly = TRUE) &&
#'     requireNamespace("tibble", quietly = TRUE)) {
#'   # Create small example
#'   mat <- matrix(rnorm(12), nrow = 3, ncol = 4)
#'   delayed_mat <- DelayedArray::DelayedArray(mat)
#'   
#'   # Create FmriSeries with metadata
#'   fs <- new("FmriSeries", delayed_mat,
#'             voxel_info = S4Vectors::DataFrame(
#'               voxel_id = 1:4,
#'               region = c("A", "A", "B", "B")
#'             ),
#'             temporal_info = S4Vectors::DataFrame(
#'               time = 1:3,
#'               condition = c("rest", "task", "rest")
#'             ),
#'             selection_info = list(),
#'             dataset_info = list())
#'   
#'   # Convert to tibble
#'   tbl_result <- tibble::as_tibble(fs)
#'   # Result has 12 rows (3 timepoints x 4 voxels)
#'   # with columns: time, condition, voxel_id, region, signal
#' }
#' }
#' 
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