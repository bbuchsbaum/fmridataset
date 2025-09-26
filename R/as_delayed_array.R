#' Convert Backend to DelayedArray
#'
#' Provides a DelayedArray interface for storage backends. The returned
#' object lazily retrieves data via the backend when subsets of the array
#' are accessed.
#'
#' @param backend A storage backend object
#' @param sparse_ok Logical, allow sparse representation when possible
#' @param ... Additional arguments passed to methods
#' @return A DelayedArray object
#' @importFrom methods setClass setMethod new setOldClass
#' @examples
#' \dontrun{
#' b <- matrix_backend(matrix(rnorm(20), nrow = 5))
#' da <- as_delayed_array(b)
#' dim(da)
#' }
#' @export
as_delayed_array <- function(backend, sparse_ok = FALSE, ...) {
  UseMethod("as_delayed_array")
}

setOldClass("matrix_backend")
setOldClass("nifti_backend")
setOldClass("study_backend")
setOldClass("study_backend_seed")
setOldClass("matrix_dataset")
setOldClass("fmri_file_dataset")
setOldClass("fmri_mem_dataset")

.delayed_array_support_env <- new.env(parent = emptyenv())
.delayed_array_support_env$registered <- FALSE

.ensure_delayed_array <- function() {
  if (isTRUE(getOption("fmridataset.disable_delayedarray", FALSE))) {
    stop(
      "DelayedArray support is disabled in this session.",
      call. = FALSE
    )
  }
  if (!.require_namespace("DelayedArray", quietly = TRUE)) {
    stop(
      "The DelayedArray package is required for as_delayed_array() operations.",
      call. = FALSE
    )
  }
}

register_delayed_array_support <- function() {
  if (isTRUE(.delayed_array_support_env$registered)) {
    return(invisible(NULL))
  }

  .ensure_delayed_array()

  methods::setClass(
    "StorageBackendSeed",
    slots = list(backend = "ANY"),
    contains = "Array"
  )

  methods::setMethod("dim", "StorageBackendSeed", function(x) {
    d <- backend_get_dims(x@backend)
    num_voxels <- sum(backend_get_mask(x@backend))
    c(d$time, num_voxels)
  })

  extract_array_generic <- getExportedValue("DelayedArray", "extract_array")

  methods::setMethod(extract_array_generic, "StorageBackendSeed", function(x, index) {
    rows <- if (length(index) >= 1) index[[1]] else NULL
    cols <- if (length(index) >= 2) index[[2]] else NULL
    backend_get_data(x@backend, rows = rows, cols = cols)
  })

  methods::setClass("NiftiBackendSeed", contains = "StorageBackendSeed")
  methods::setClass("MatrixBackendSeed", contains = "StorageBackendSeed")

  register_study_backend_seed_methods()

  .delayed_array_support_env$registered <- TRUE
  invisible(NULL)
}

#' @rdname as_delayed_array
#' @method as_delayed_array nifti_backend
#' @export
as_delayed_array.nifti_backend <- function(backend, sparse_ok = FALSE, ...) {
  .ensure_delayed_array()
  register_delayed_array_support()
  seed <- methods::new("NiftiBackendSeed", backend = backend)
  getExportedValue("DelayedArray", "DelayedArray")(seed)
}

#' @rdname as_delayed_array
#' @method as_delayed_array matrix_backend
#' @export
as_delayed_array.matrix_backend <- function(backend, sparse_ok = FALSE, ...) {
  .ensure_delayed_array()
  register_delayed_array_support()
  seed <- methods::new("MatrixBackendSeed", backend = backend)
  getExportedValue("DelayedArray", "DelayedArray")(seed)
}

#' @rdname as_delayed_array
#' @method as_delayed_array study_backend
#' @export
as_delayed_array.study_backend <- function(backend, sparse_ok = FALSE, ...) {
  .ensure_delayed_array()
  register_delayed_array_support()
  seed <- study_backend_seed(
    backends = backend$backends,
    subject_ids = backend$subject_ids
  )
  getExportedValue("DelayedArray", "DelayedArray")(seed)
}

#' @rdname as_delayed_array
#' @method as_delayed_array default
#' @export
as_delayed_array.default <- function(backend, sparse_ok = FALSE, ...) {
  stop("No as_delayed_array method for class: ", class(backend)[1])
}
