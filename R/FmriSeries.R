#' fmri_series: fMRI Time Series Container
#'
#' @description
#' An S3 class representing lazily accessed fMRI time series data. The
#' underlying data is stored in a `DelayedMatrix` with rows corresponding
#' to timepoints and columns corresponding to voxels.
#'
#' @details
#' An fmri_series object contains:
#' - `data`: A DelayedMatrix with timepoints as rows and voxels as columns
#' - `voxel_info`: A data.frame containing spatial metadata for each voxel
#' - `temporal_info`: A data.frame containing metadata for each timepoint
#' - `selection_info`: A list describing how the data were selected
#' - `dataset_info`: A list describing the source dataset and backend
#'
#' @return An object of class \code{fmri_series}
#'
#' @seealso
#' \code{\link{as.matrix.fmri_series}} for converting to standard matrix,
#' \code{\link{as_tibble.fmri_series}} for converting to tibble format
#'
#' @examples
#' \donttest{
#' # Create example fmri_series object
#' # Create small example data
#' mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
#' delayed_mat <- DelayedArray::DelayedArray(mat)
#'
#' # Create metadata
#' vox_info <- data.frame(
#'   x = rep(1:10, 5),
#'   y = rep(1:5, each = 10),
#'   z = 1
#' )
#'
#' temp_info <- data.frame(
#'   time = seq(0, 99, by = 1),
#'   run = rep(1:4, each = 25)
#' )
#'
#' # Create fmri_series object
#' fs <- new_fmri_series(
#'   data = delayed_mat,
#'   voxel_info = vox_info,
#'   temporal_info = temp_info,
#'   selection_info = list(),
#'   dataset_info = list()
#' )
#' }
#'
#' @name fmri_series
NULL

#' Constructor for fmri_series objects
#'
#' @param data A DelayedMatrix with timepoints as rows and voxels as columns
#' @param voxel_info A data.frame containing spatial metadata for each voxel
#' @param temporal_info A data.frame containing metadata for each timepoint
#' @param selection_info A list describing how the data were selected
#' @param dataset_info A list describing the source dataset and backend
#'
#' @return An object of class \code{fmri_series}
#' @keywords internal
new_fmri_series <- function(data, voxel_info, temporal_info, selection_info, dataset_info) {
  stopifnot(inherits(data, "DelayedMatrix") || is.matrix(data))
  stopifnot(is.data.frame(voxel_info))
  stopifnot(is.data.frame(temporal_info))
  stopifnot(is.list(selection_info))
  stopifnot(is.list(dataset_info))

  # Ensure dimensions match
  stopifnot(nrow(voxel_info) == ncol(data))
  stopifnot(nrow(temporal_info) == nrow(data))

  structure(
    list(
      data = data,
      voxel_info = voxel_info,
      temporal_info = temporal_info,
      selection_info = selection_info,
      dataset_info = dataset_info
    ),
    class = "fmri_series"
  )
}

#' Print Method for fmri_series Objects
#'
#' @description
#' Display a concise summary of an fmri_series object, including dimensions,
#' selector type, backend, and data orientation.
#'
#' @param x An \code{fmri_series} object
#' @param ... Additional arguments (unused)
#'
#' @return Returns \code{x} invisibly. Called for its side effect of
#'   printing to the console.
#'
#' @examples
#' \donttest{
#' # This method is called automatically when printing
#' # Create example object (see fmri_series documentation)
#' # fs <- new_fmri_series(...)
#' # fs  # Automatically calls print method
#' }
#'
#' @export
print.fmri_series <- function(x, ...) {
  n_time <- nrow(x$data)
  n_vox <- ncol(x$data)
  cat(sprintf(
    "<fmri_series> %s voxels × %s timepoints (lazy)\n",
    n_vox, n_time
  ))
  sel <- x$selection_info
  dataset <- x$dataset_info
  sel_desc <- if (!is.null(sel$selector)) "custom" else "NULL"
  backend <- if (!is.null(dataset$backend_type)) dataset$backend_type else "?"
  cat(sprintf(
    "Selector: %s | Backend: %s | Orientation: time × voxels\n",
    sel_desc, backend
  ))
  invisible(x)
}

#' Convert fmri_series to Matrix
#'
#' @description
#' This method realizes the underlying DelayedMatrix and
#' returns an ordinary matrix with timepoints in rows and
#' voxels in columns.
#'
#' @param x An \code{fmri_series} object
#' @param ... Additional arguments (ignored)
#'
#' @return A matrix with timepoints as rows and voxels as columns
#'
#' @seealso
#' \code{\link{fmri_series}} for the class definition,
#' \code{\link{as_tibble.fmri_series}} for tibble conversion
#'
#' @examples
#' \donttest{
#' # Create small example
#' mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
#' delayed_mat <- DelayedArray::DelayedArray(mat)
#'
#' # Create minimal fmri_series object
#' fs <- new_fmri_series(
#'   data = delayed_mat,
#'   voxel_info = data.frame(idx = 1:5),
#'   temporal_info = data.frame(time = 1:4),
#'   selection_info = list(),
#'   dataset_info = list()
#' )
#'
#' # Convert to matrix
#' mat_result <- as.matrix(fs)
#' dim(mat_result) # 4 x 5
#' }
#'
#' @export
as.matrix.fmri_series <- function(x, ...) {
  if (inherits(x$data, "DelayedMatrix")) {
    DelayedArray::as.matrix(x$data)
  } else {
    as.matrix(x$data)
  }
}

#' Convert fmri_series to Tibble
#'
#' @description
#' The returned tibble contains one row per voxel/timepoint
#' combination with metadata columns from \code{temporal_info}
#' and \code{voxel_info} and a \code{signal} column with the data
#' values.
#'
#' @param x An \code{fmri_series} object
#' @param ... Additional arguments (ignored)
#'
#' @return A tibble with columns from temporal_info, voxel_info, and a
#'   signal column containing the fMRI signal values
#'
#' @seealso
#' \code{\link{fmri_series}} for the class definition,
#' \code{\link{as.matrix.fmri_series}} for matrix conversion
#'
#' @examples
#' \donttest{
#' # Create small example
#' mat <- matrix(rnorm(12), nrow = 3, ncol = 4)
#' delayed_mat <- DelayedArray::DelayedArray(mat)
#'
#' # Create fmri_series with metadata
#' fs <- new_fmri_series(
#'   data = delayed_mat,
#'   voxel_info = data.frame(
#'     voxel_id = 1:4,
#'     region = c("A", "A", "B", "B")
#'   ),
#'   temporal_info = data.frame(
#'     time = 1:3,
#'     condition = c("rest", "task", "rest")
#'   ),
#'   selection_info = list(),
#'   dataset_info = list()
#' )
#'
#' # Convert to tibble
#' tbl_result <- tibble::as_tibble(fs)
#' # Result has 12 rows (3 timepoints x 4 voxels)
#' # with columns: time, condition, voxel_id, region, signal
#' }
#'
#' @export
#' @importFrom tibble as_tibble
as_tibble.fmri_series <- function(x, ...) {
  mat <- as.matrix(x)
  vox_df <- x$voxel_info
  tmp_df <- x$temporal_info

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

#' Check if object is an fmri_series
#'
#' @param x An object to test
#' @return Logical TRUE if x is an fmri_series object
#' @export
is.fmri_series <- function(x) {
  inherits(x, "fmri_series")
}

#' Dimensions of fmri_series
#'
#' @param x An fmri_series object
#' @return Integer vector of length 2 (timepoints, voxels)
#' @method dim fmri_series
#' @export
dim.fmri_series <- function(x) {
  dim(x$data)
}

#' Number of rows in fmri_series
#'
#' @param x An fmri_series object
#' @return Number of timepoints
#' @method nrow fmri_series
#' @export
nrow.fmri_series <- function(x) {
  nrow(x$data)
}

#' Number of columns in fmri_series
#'
#' @param x An fmri_series object
#' @return Number of voxels
#' @method ncol fmri_series
#' @export
ncol.fmri_series <- function(x) {
  ncol(x$data)
}
