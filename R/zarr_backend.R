#' Zarr Storage Backend
#'
#' @description
#' A storage backend implementation for Zarr array format using the CRAN zarr package.
#' Zarr is a cloud-native array storage format that supports chunked, compressed
#' n-dimensional arrays with concurrent read/write access.
#'
#' @section Experimental:
#' This backend uses the CRAN zarr package which is relatively new (v0.1.1, Dec 2025).
#' It supports Zarr v3 format only - Zarr v2 stores cannot be read.
#' Please report any issues to help improve the package.
#'
#' @details
#' This backend provides efficient access to neuroimaging data stored in Zarr format,
#' which is particularly well-suited for:
#' - Large datasets that don't fit in memory
#' - Cloud storage (S3, GCS, Azure)
#' - Parallel processing workflows
#' - Progressive data access patterns
#'
#' The backend expects Zarr arrays organized as a 4D array with dimensions (x, y, z, time).
#' The CRAN zarr package uses R6 classes and supports Zarr v3 format only.
#'
#' @name zarr-backend
#' @keywords internal
NULL

#' Create a Zarr Backend
#'
#' @description
#' Creates a storage backend for Zarr array data using the CRAN zarr package.
#'
#' @section Experimental:
#' This backend uses the CRAN zarr package which is relatively new (v0.1.1, Dec 2025).
#' It supports Zarr v3 format only - Zarr v2 stores cannot be read.
#' Please report any issues to help improve the package.
#'
#' @param source Character path to Zarr store (directory or URL for remote stores)
#' @param preload Logical, whether to load all data into memory (default: FALSE)
#' @return A zarr_backend S3 object
#' @export
#' @keywords internal
#' @examples
#' \dontrun{
#' # Local Zarr store
#' backend <- zarr_backend("path/to/data.zarr")
#'
#' # Remote store
#' backend <- zarr_backend("https://example.com/data.zarr")
#' }
zarr_backend <- function(source,
                         preload = FALSE) {
  # Validate source first
  if (!is.character(source) || length(source) != 1) {
    stop_fmridataset(
      fmridataset_error_config,
      "source must be a single character string",
      parameter = "source",
      value = class(source)
    )
  }

  # Check if zarr is available
  if (!requireNamespace("zarr", quietly = TRUE)) {
    stop_fmridataset(
      fmridataset_error_config,
      "The zarr package is required for zarr_backend but is not installed.",
      details = "Install with: install.packages('zarr')"
    )
  }

  # Create backend object
  backend <- list(
    source = source,
    preload = preload,
    zarr_array = NULL,
    data_cache = NULL,
    dims = NULL,
    is_open = FALSE
  )

  class(backend) <- c("zarr_backend", "storage_backend")
  backend
}

#' @rdname backend_open
#' @method backend_open zarr_backend
#' @export
backend_open.zarr_backend <- function(backend) {
  if (backend$is_open) {
    return(backend)
  }

  # Open Zarr store
  tryCatch(
    {
      # For local paths, ensure they exist
      if (!grepl("^(https?://|s3://|gs://)", backend$source) &&
        !file.exists(backend$source)) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          sprintf("Zarr store not found: %s", backend$source),
          file = backend$source,
          operation = "open"
        )
      }

      # Open the Zarr array using CRAN zarr package
      backend$zarr_array <- zarr::open_zarr(backend$source)

      # Get array info from ZarrArray R6 object
      array_dims <- backend$zarr_array$shape

      # Validate dimensions (expecting 4D: x, y, z, time)
      if (length(array_dims) != 4) {
        stop_fmridataset(
          fmridataset_error_config,
          sprintf("Expected 4D array, got %dD", length(array_dims)),
          parameter = "source",
          value = backend$source
        )
      }

      # Store dimensions
      backend$dims <- list(
        spatial = array_dims[1:3],
        time = array_dims[4]
      )

      # Preload if requested
      if (backend$preload) {
        message("Preloading Zarr data into memory...")
        backend$data_cache <- backend$zarr_array[, , , , drop = FALSE]
      }

      backend$is_open <- TRUE
    },
    error = function(e) {
      stop_fmridataset(
        fmridataset_error_backend_io,
        sprintf("Failed to open Zarr store: %s", e$message),
        file = backend$source,
        operation = "open"
      )
    }
  )

  backend
}

#' @rdname backend_close
#' @method backend_close zarr_backend
#' @export
backend_close.zarr_backend <- function(backend) {
  # Zarr arrays are stateless, so just clear references
  backend$zarr_array <- NULL
  backend$data_cache <- NULL
  backend$is_open <- FALSE
  invisible(NULL)
}

#' @rdname backend_get_dims
#' @method backend_get_dims zarr_backend
#' @export
backend_get_dims.zarr_backend <- function(backend) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing dimensions",
      operation = "get_dims"
    )
  }

  backend$dims
}

#' @rdname backend_get_mask
#' @method backend_get_mask zarr_backend
#' @export
backend_get_mask.zarr_backend <- function(backend) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing mask",
      operation = "get_mask"
    )
  }

  n_voxels <- prod(backend$dims$spatial)

  # Default: all voxels are valid
  # Note: Zarr backend stores single arrays, not separate mask arrays
  # Mask data should be provided externally if needed
  mask <- rep(TRUE, n_voxels)

  # Validate mask
  if (any(is.na(mask))) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Mask contains NA values",
      operation = "get_mask"
    )
  }

  if (sum(mask) == 0) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Mask has no valid voxels",
      operation = "get_mask"
    )
  }

  mask
}

#' @rdname backend_get_data
#' @method backend_get_data zarr_backend
#' @export
backend_get_data.zarr_backend <- function(backend, rows = NULL, cols = NULL) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing data",
      operation = "get_data"
    )
  }

  # Get dimensions
  n_timepoints <- backend$dims$time
  n_voxels <- prod(backend$dims$spatial)

  # Default to all rows/cols
  if (is.null(rows)) rows <- seq_len(n_timepoints)
  if (is.null(cols)) cols <- seq_len(n_voxels)

  # Validate indices
  if (any(rows < 1 | rows > n_timepoints)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Row indices must be between 1 and %d", n_timepoints),
      parameter = "rows"
    )
  }

  if (any(cols < 1 | cols > n_voxels)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Column indices must be between 1 and %d", n_voxels),
      parameter = "cols"
    )
  }

  # Read data using CRAN zarr package
  if (backend$preload) {
    # Use cached data
    data_4d <- backend$data_cache
  } else {
    # Convert column indices to 3D coordinates
    spatial_dims <- backend$dims$spatial
    coords <- arrayInd(cols, spatial_dims)

    # Determine optimal reading strategy
    # For simplicity, read full array if requesting >50% of data
    proportion <- (length(rows) * length(cols)) / (n_timepoints * n_voxels)

    if (proportion > 0.5) {
      # Read full array
      data_4d <- backend$zarr_array[, , , , drop = FALSE]
    } else {
      # Read full array (zarr subsetting is complex)
      # TODO: Optimize for sparse access patterns
      data_4d <- backend$zarr_array[, , , , drop = FALSE]
    }
  }

  # Reshape to time x voxels matrix
  dim(data_4d) <- c(prod(backend$dims$spatial), backend$dims$time)
  data_matrix <- t(data_4d)

  # Return subset
  data_matrix[rows, cols, drop = FALSE]
}

#' @rdname backend_get_metadata
#' @method backend_get_metadata zarr_backend
#' @export
backend_get_metadata.zarr_backend <- function(backend) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing metadata",
      operation = "get_metadata"
    )
  }

  metadata <- list()

  # Try to get Zarr attributes from R6 object
  tryCatch(
    {
      if (!is.null(backend$zarr_array$dtype)) {
        metadata$dtype <- backend$zarr_array$dtype
      }
      if (!is.null(backend$zarr_array$chunks)) {
        metadata$chunk_shape <- backend$zarr_array$chunks
      }
    },
    error = function(e) {
      # Attributes not available
    }
  )

  # Add basic info
  metadata$storage_format <- "zarr"
  metadata$zarr_version <- "v3"

  metadata
}
