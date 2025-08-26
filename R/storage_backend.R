#' Storage Backend S3 Contract
#'
#' @description
#' Defines the S3 generic functions that all storage backends must implement.
#' This provides a pluggable architecture for different data storage formats.
#'
#' @details
#' A storage backend is responsible for:
#' - Managing stateful resources (file handles, connections)
#' - Providing dimension information
#' - Reading data in canonical timepoints × voxels orientation
#' - Providing mask information
#' - Extracting metadata
#'
#' @name storage-backend
#' @keywords internal
NULL

#' Open a Storage Backend
#'
#' @description
#' Opens a storage backend and acquires any necessary resources (e.g., file handles).
#' Stateless backends can implement this as a no-op.
#'
#' @param backend A storage backend object
#' @return The backend object (possibly modified with state)
#' @export
#' @keywords internal
backend_open <- function(backend) {
  UseMethod("backend_open")
}

#' Close a Storage Backend
#'
#' @description
#' Closes a storage backend and releases any resources.
#' Stateless backends can implement this as a no-op.
#'
#' @param backend A storage backend object
#' @return NULL (invisibly)
#' @export
#' @keywords internal
backend_close <- function(backend) {
  UseMethod("backend_close")
}

#' Get Dimensions from Backend
#'
#' @description
#' Returns the dimensions of the data stored in the backend.
#'
#' @param backend A storage backend object
#' @return A named list with elements:
#'   - spatial: numeric vector of length 3 (x, y, z dimensions)
#'   - time: integer, number of timepoints
#' @export
#' @keywords internal
backend_get_dims <- function(backend) {
  UseMethod("backend_get_dims")
}

#' Get Mask from Backend
#'
#' @description
#' Returns a logical mask indicating which voxels contain valid data.
#'
#' @param backend A storage backend object
#' @return A logical vector satisfying:
#'   - length(mask) == prod(backend_get_dims(backend)$spatial)
#'   - sum(mask) > 0 (no empty masks allowed)
#'   - No NA values allowed
#' @export
#' @keywords internal
backend_get_mask <- function(backend) {
  UseMethod("backend_get_mask")
}

#' Get Data from Backend
#'
#' @description
#' Reads data from the backend in canonical timepoints × voxels orientation.
#'
#' @param backend A storage backend object
#' @param rows Integer vector of row indices (timepoints) to read, or NULL for all
#' @param cols Integer vector of column indices (voxels) to read, or NULL for all
#' @return A matrix in timepoints × voxels orientation
#' @export
#' @keywords internal
backend_get_data <- function(backend, rows = NULL, cols = NULL) {
  UseMethod("backend_get_data")
}

#' Get Metadata from Backend
#'
#' @description
#' Returns metadata associated with the data (e.g., affine matrix, voxel dimensions).
#'
#' @param backend A storage backend object
#' @return A list containing neuroimaging metadata, which may include:
#'   - affine: 4x4 affine transformation matrix
#'   - voxel_dims: numeric vector of voxel dimensions
#'   - intent_code: NIfTI intent code
#'   - Additional format-specific metadata
#' @export
#' @keywords internal
backend_get_metadata <- function(backend) {
  UseMethod("backend_get_metadata")
}

#' Validate Backend Implementation
#'
#' @description
#' Validates that a backend implements the required contract correctly.
#'
#' @param backend A storage backend object
#' @return TRUE if valid, otherwise throws an error
#' @keywords internal
validate_backend <- function(backend) {
  # First check basic inheritance
  if (!inherits(backend, "storage_backend")) {
    stop_fmridataset(
      fmridataset_error_config,
      "Invalid backend object: must inherit from 'storage_backend'"
    )
  }

  # Check that required methods are implemented
  backend_class <- class(backend)[1]
  required_methods <- c(
    "backend_open", "backend_close", "backend_get_dims",
    "backend_get_mask", "backend_get_data", "backend_get_metadata"
  )

  for (method in required_methods) {
    method_name <- paste0(method, ".", backend_class)
    if (!exists(method_name, mode = "function")) {
      stop_fmridataset(
        fmridataset_error_config,
        sprintf("Backend class '%s' must implement method '%s'", backend_class, method)
      )
    }
  }

  backend <- backend_open(backend)
  on.exit(backend_close(backend))

  dims <- backend_get_dims(backend)
  if (!is.list(dims) || !all(c("spatial", "time") %in% names(dims))) {
    stop_fmridataset(
      fmridataset_error_config,
      "backend_get_dims must return a list with 'spatial' and 'time' elements"
    )
  }

  if (length(dims$spatial) != 3 || !is.numeric(dims$spatial)) {
    stop_fmridataset(
      fmridataset_error_config,
      "spatial dimensions must be a numeric vector of length 3"
    )
  }

  if (!is.numeric(dims$time) || length(dims$time) != 1 || dims$time < 1) {
    stop_fmridataset(
      fmridataset_error_config,
      "time dimension must be a positive integer"
    )
  }

  mask <- backend_get_mask(backend)
  expected_length <- prod(dims$spatial)

  if (!is.logical(mask)) {
    stop_fmridataset(
      fmridataset_error_config,
      "backend_get_mask must return a logical vector"
    )
  }

  if (length(mask) != expected_length) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf(
        "mask length (%d) must equal prod(spatial dims) (%d)",
        length(mask), expected_length
      )
    )
  }

  if (sum(mask) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      "mask must contain at least one TRUE value"
    )
  }

  if (any(is.na(mask))) {
    stop_fmridataset(
      fmridataset_error_config,
      "mask cannot contain NA values"
    )
  }

  TRUE
}
