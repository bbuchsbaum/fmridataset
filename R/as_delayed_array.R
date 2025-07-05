#' Convert Backend to DelayedArray
#'
#' Provides a DelayedArray interface for storage backends. The returned
#' object lazily retrieves data via the backend when subsets of the array
#' are accessed.
#'
#' @param backend A storage backend object
#' @param sparse_ok Logical, allow sparse representation when possible
#' @return A DelayedArray object
#' @examples
#' \dontrun{
#'   b <- matrix_backend(matrix(rnorm(20), nrow = 5))
#'   da <- as_delayed_array(b)
#'   dim(da)
#' }
#' @importFrom DelayedArray extract_array DelayedArray
#' @importFrom methods setClass setMethod setGeneric new
#' @export
setGeneric("as_delayed_array", function(backend, sparse_ok = FALSE)
    standardGeneric("as_delayed_array"))

# Base seed class ---------------------------------------------------------

setClass("StorageBackendSeed",
         slots = list(backend = "ANY"),
         contains = "Array")

#' Dimensions of StorageBackendSeed
#' 
#' @param x A StorageBackendSeed object
#' @return An integer vector of length 2 (timepoints, voxels)
#' @rdname dim-StorageBackendSeed-method
#' @aliases dim,StorageBackendSeed-method
#' @keywords internal
setMethod("dim", "StorageBackendSeed", function(x) {
  d <- backend_get_dims(x@backend)
  # The number of voxels is the number of TRUE values in the mask
  num_voxels <- sum(backend_get_mask(x@backend))
  c(d$time, num_voxels)
})

#' @keywords internal
setMethod("extract_array", "StorageBackendSeed", function(x, index) {
  rows <- if (length(index) >= 1) index[[1]] else NULL
  cols <- if (length(index) >= 2) index[[2]] else NULL
  backend_get_data(x@backend, rows = rows, cols = cols)
})

# Specific seeds ---------------------------------------------------------

setClass("NiftiBackendSeed", contains = "StorageBackendSeed")
setClass("MatrixBackendSeed", contains = "StorageBackendSeed")

# Methods for backends ---------------------------------------------------

#' @rdname as_delayed_array
#' @aliases as_delayed_array,nifti_backend-method
setMethod("as_delayed_array", "nifti_backend", function(backend, sparse_ok = FALSE) {
  seed <- new("NiftiBackendSeed", backend = backend)
  DelayedArray::DelayedArray(seed)
})

#' @rdname as_delayed_array
#' @aliases as_delayed_array,matrix_backend-method
setMethod("as_delayed_array", "matrix_backend", function(backend, sparse_ok = FALSE) {
  seed <- new("MatrixBackendSeed", backend = backend)
  DelayedArray::DelayedArray(seed)
})

#' @rdname as_delayed_array
#' @aliases as_delayed_array,study_backend-method
setMethod("as_delayed_array", "study_backend", function(backend, sparse_ok = FALSE) {
  # Use the new StudyBackendSeed for true lazy evaluation
  seed <- StudyBackendSeed(
    backends = backend$backends,
    subject_ids = backend$subject_ids
  )
  
  DelayedArray::DelayedArray(seed)
})
