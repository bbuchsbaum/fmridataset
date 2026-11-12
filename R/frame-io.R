# Indirections so tests can exercise the missing-package error path whether or
# not fmristore is installed.
.frame_store_available <- function() requireNamespace("fmristore", quietly = TRUE)

.frame_store_exports <- function() getNamespaceExports("fmristore")

.require_frame_store <- function(function_name) {
  if (!.frame_store_available()) {
    .frame_abort(
      paste(
        "HDF5 frame I/O requires the fmristore package, which is not on CRAN;",
        "install it from https://bbuchsbaum.r-universe.dev."
      ),
      "fmridataset_error_backend_io",
      operation = function_name
    )
  }
  if (!function_name %in% .frame_store_exports()) {
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

# fmristore releases that predate the renames in fmridataset-deprecated still
# call the old names. A user of write_frame()/open_frame() cannot act on those
# warnings, so they are muffled on this path only; remove this with the
# deprecated aliases.
.muffle_store_deprecations <- function(expr) {
  withCallingHandlers(
    expr,
    deprecatedWarning = function(w) {
      if (grepl("fmridataset-deprecated", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
}

#' Persist and reopen an fmri frame
#'
#' `write_frame()` writes an `fmri_frame` to an HDF5 file and `open_frame()`
#' reopens one. Both require the `fmristore` package, which performs the
#' physical HDF5 work. `fmristore` is not on CRAN; install it from
#' <https://bbuchsbaum.r-universe.dev>, for example with
#' `install.packages("fmristore", repos = c("https://bbuchsbaum.r-universe.dev",
#' getOption("repos")))`. Without it, or with a version that lacks frame
#' support, both functions fail with an error of class
#' `fmridataset_error_backend_io`.
#'
#' Reopened assays are lazy sources: opening a frame does not read assay
#' values.
#'
#' Neither function computes a content hash. Persistence records semantic
#' manifest digests and source fingerprints only; a caller who wants a value
#' receipt for the written or reopened arrays requests it explicitly with
#' [content_hash()] and records the result where it is needed.
#'
#' @param x An `fmri_frame`.
#' @param path Destination or source path.
#' @param format Storage format. Only `"hdf5"` is supported.
#' @param ... Arguments passed to the physical store implementation.
#' @return `write_frame()` invisibly returns the committed path, normalized
#'   with forward slashes on every platform. `open_frame()` returns an
#'   `fmri_frame`.
#' @examples
#' if (requireNamespace("fmristore", quietly = TRUE)) {
#'   src <- memory_source(matrix(seq_len(6), nrow = 2))
#'   obs <- tibble::tibble(.obs_id = c("o1", "o2"))
#'   space <- index_space(3, id_policy = "deterministic", namespace = "demo")
#'   frame <- fmri_frame(list(beta = src), obs, space = space)
#'   path <- tempfile(fileext = ".h5")
#'   committed <- write_frame(frame, path)
#'   reopened <- open_frame(committed)
#'   collect_assay(reopened, "beta")
#' }
#' @export
write_frame <- function(x, path, format = "hdf5", ...) {
  format <- match.arg(format, "hdf5")
  if (inherits(x, c("fmri_study", "fmri_collection"))) {
    .frame_abort(
      paste0(
        "write_frame() saves one frame, not an ", class(x)[[1L]], ".\n",
        "  Write each frame with write_frame() and keep the description from\n",
        "  fds_study_manifest(); study_from_fds_manifest() rebuilds the study.\n",
        "  See vignette(\"persistence-and-import\")."
      ),
      "fmridataset_error_backend_io",
      operation = "write_frame", actual = class(x)
    )
  }
  # Refuse a frame that cannot be persisted (ephemeral IDs) before asking for
  # the storage package, so the error is about the frame, not the setup.
  fds_frame_manifest(x)
  .require_frame_store("write_frame_h5")
  # Evaluate the caller's arguments first so that only warnings raised inside
  # the store are muffled, never ones from the caller's own expressions.
  force(x)
  force(path)
  extra <- list(...)
  committed <- .muffle_store_deprecations(
    do.call(fmristore::write_frame_h5, c(list(x, path), extra))
  )
  invisible(normalizePath(committed, winslash = "/", mustWork = TRUE))
}

#' @rdname write_frame
#' @export
open_frame <- function(path, format = "hdf5", ...) {
  format <- match.arg(format, "hdf5")
  .require_frame_store("open_frame_h5")
  force(path)
  extra <- list(...)
  .muffle_store_deprecations(do.call(fmristore::open_frame_h5, c(list(path), extra)))
}
