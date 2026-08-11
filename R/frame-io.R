.require_frame_store <- function(function_name) {
  if (!requireNamespace("fmristore", quietly = TRUE)) {
    .frame_abort(
      "HDF5 frame I/O requires the optional fmristore package.",
      "fmridataset_error_backend_io",
      operation = function_name
    )
  }
  if (!function_name %in% getNamespaceExports("fmristore")) {
    .frame_abort(
      paste0(
        "The installed fmristore does not provide `",
        function_name,
        "()`; install a version with FDS frame support."
      ),
      "fmridataset_error_backend_io",
      operation = function_name
    )
  }
  invisible(TRUE)
}

#' Persist and reopen an fmri frame
#'
#' These functions provide the semantic-package entry point while delegating
#' physical HDF5 work to `fmristore`. Reopened assays are reconstructible lazy
#' sources; opening a frame does not read assay values.
#'
#' @param x An `fmri_frame`.
#' @param path Destination or source path.
#' @param format Storage format. The walking-skeleton implementation supports
#'   `"hdf5"`.
#' @param ... Arguments passed to the physical store implementation.
#' @return `write_frame()` invisibly returns the committed path. `open_frame()`
#'   returns an `fmri_frame`.
#' @export
write_frame <- function(x, path, format = "hdf5", ...) {
  format <- match.arg(format, "hdf5")
  .require_frame_store("write_frame_h5")
  fmristore::write_frame_h5(x, path, ...)
}

#' @rdname write_frame
#' @export
open_frame <- function(path, format = "hdf5", ...) {
  format <- match.arg(format, "hdf5")
  .require_frame_store("open_frame_h5")
  fmristore::open_frame_h5(path, ...)
}
