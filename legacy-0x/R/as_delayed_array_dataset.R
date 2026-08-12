#' Convert Dataset Objects to DelayedArray
#'
#' Provides DelayedArray interface for dataset objects. These methods
#' convert fmri_dataset and matrix_dataset objects to DelayedArrays
#' for memory-efficient operations.
#'
#' @name as_delayed_array-dataset
#' @importFrom methods setMethod
#' @inheritParams as_delayed_array
NULL

#' @rdname as_delayed_array-dataset
#' @aliases as_delayed_array,matrix_dataset-method
#' @export
setMethod("as_delayed_array", "matrix_dataset", function(backend, sparse_ok = FALSE) {
  .ensure_delayed_array()
  register_delayed_array_support()
  # For matrix_dataset, the data is in memory so we can just wrap it
  # Create a matrix backend from the data
  mb <- matrix_backend(backend$datamat, mask = backend$mask > 0)
  seed <- new("MatrixBackendSeed", backend = mb)
  getExportedValue("DelayedArray", "DelayedArray")(seed)
})

#' @rdname as_delayed_array-dataset
#' @aliases as_delayed_array,fmri_file_dataset-method
#' @export
setMethod("as_delayed_array", "fmri_file_dataset", function(backend, sparse_ok = FALSE) {
  .ensure_delayed_array()
  register_delayed_array_support()
  # Use the backend if available
  if (!is.null(backend$backend)) {
    as_delayed_array(backend$backend, sparse_ok = sparse_ok)
  } else {
    # Legacy path - need to handle differently
    stop("as_delayed_array not supported for legacy fmri_file_dataset objects")
  }
})

#' @rdname as_delayed_array-dataset
#' @aliases as_delayed_array,fmri_mem_dataset-method
#' @export
setMethod("as_delayed_array", "fmri_mem_dataset", function(backend, sparse_ok = FALSE) {
  .ensure_delayed_array()
  register_delayed_array_support()
  # For memory datasets, get the data matrix and create a backend
  mat <- get_data_matrix(backend)
  mb <- matrix_backend(mat, mask = backend$mask > 0)
  seed <- new("MatrixBackendSeed", backend = mb)
  getExportedValue("DelayedArray", "DelayedArray")(seed)
})
