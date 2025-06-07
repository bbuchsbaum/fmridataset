#' Matrix Storage Backend
#'
#' @description
#' A storage backend implementation for in-memory matrix data.
#' This backend wraps existing matrix data in the storage backend interface.
#'
#' @name matrix-backend
#' @keywords internal
NULL

#' Create a Matrix Backend
#'
#' @param data_matrix A matrix in timepoints × voxels orientation
#' @param mask Logical vector indicating which voxels are valid
#' @param spatial_dims Numeric vector of length 3 specifying spatial dimensions
#' @param metadata Optional list of metadata
#' @return A matrix_backend S3 object
#' @export
#' @keywords internal
matrix_backend <- function(data_matrix, mask = NULL, spatial_dims = NULL, metadata = NULL) {
  # Validate inputs
  if (!is.matrix(data_matrix)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "data_matrix must be a matrix",
      parameter = "data_matrix",
      value = class(data_matrix)
    )
  }

  n_timepoints <- nrow(data_matrix)
  n_voxels <- ncol(data_matrix)

  # Default mask: all voxels are valid
  if (is.null(mask)) {
    mask <- rep(TRUE, n_voxels)
  }

  # Validate mask
  if (!is.logical(mask)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "mask must be a logical vector",
      parameter = "mask",
      value = class(mask)
    )
  }

  if (length(mask) != n_voxels) {
    stop_fmridataset(
      fmridataset_error_config,
      message = sprintf(
        "mask length (%d) must equal number of columns (%d)",
        length(mask), n_voxels
      ),
      parameter = "mask"
    )
  }

  # Default spatial dimensions: try to factorize n_voxels
  if (is.null(spatial_dims)) {
    # Simple approach: create a "flat" 3D volume
    spatial_dims <- c(n_voxels, 1, 1)
  }

  # Validate spatial dimensions
  if (length(spatial_dims) != 3 || !is.numeric(spatial_dims)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "spatial_dims must be a numeric vector of length 3",
      parameter = "spatial_dims",
      value = spatial_dims
    )
  }

  if (prod(spatial_dims) != n_voxels) {
    stop_fmridataset(
      fmridataset_error_config,
      message = sprintf(
        "Product of spatial_dims (%d) must equal number of voxels (%d)",
        prod(spatial_dims), n_voxels
      ),
      parameter = "spatial_dims"
    )
  }

  backend <- list(
    data_matrix = data_matrix,
    mask = mask,
    spatial_dims = spatial_dims,
    metadata = metadata %||% list()
  )

  class(backend) <- c("matrix_backend", "storage_backend")
  backend
}

#' @export
backend_open.matrix_backend <- function(backend) {
  # Matrix backend is stateless - no resources to acquire
  backend
}

#' @export
backend_close.matrix_backend <- function(backend) {
  # Matrix backend is stateless - no resources to release
  invisible(NULL)
}

#' @export
backend_get_dims.matrix_backend <- function(backend) {
  list(
    spatial = backend$spatial_dims,
    time = nrow(backend$data_matrix)
  )
}

#' @export
backend_get_mask.matrix_backend <- function(backend) {
  backend$mask
}

#' @export
backend_get_data.matrix_backend <- function(backend, rows = NULL, cols = NULL) {
  data <- backend$data_matrix

  # Apply subsetting if requested
  if (!is.null(rows)) {
    data <- data[rows, , drop = FALSE]
  }

  if (!is.null(cols)) {
    data <- data[, cols, drop = FALSE]
  }

  data
}

#' @export
backend_get_metadata.matrix_backend <- function(backend) {
  backend$metadata
}

# Helper function
`%||%` <- function(x, y) if (is.null(x)) y else x
