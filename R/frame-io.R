.require_frame_store <- function(function_name) {
  implementation <- .optional_export("fmristore", function_name)
  if (is.null(implementation)) {
    .frame_abort(
      paste0(
        "HDF5 frame I/O requires a compatible fmristore installation with `",
        function_name,
        "()`; fmristore is certified separately and is not a core dependency."
      ),
      "fmridataset_error_backend_io",
      operation = function_name
    )
  }
  implementation
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
  writer <- .require_frame_store("write_frame_h5")
  writer(x, path, ...)
}

#' @rdname write_frame
#' @export
open_frame <- function(path, format = "hdf5", ...) {
  format <- match.arg(format, "hdf5")
  reader <- .require_frame_store("open_frame_h5")
  reader(path, ...)
}
