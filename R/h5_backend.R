#' H5 Storage Backend
#'
#' @description
#' A storage backend implementation for H5 format neuroimaging data using fmristore.
#' Each scan is stored as an H5 file that loads to an H5NeuroVec object.
#'
#' @details
#' The H5Backend integrates with the fmristore package to work with:
#' - File paths to H5 neuroimaging files
#' - Pre-loaded H5NeuroVec objects from fmristore
#' - Multiple H5 files representing different scans
#'
#' @name h5-backend
#' @keywords internal
#' @importFrom neuroim2 space trans spacing origin series
NULL

#' Create an H5 Backend
#'
#' @param source Character vector of file paths to H5 files or list of H5NeuroVec objects
#' @param mask_source File path to H5 mask file, H5 file containing mask, or in-memory NeuroVol object
#' @param mask_dataset Character string specifying the dataset path within H5 file for mask (default: "data/elements")
#' @param data_dataset Character string specifying the dataset path within H5 files for data (default: "data")
#' @param preload Logical, whether to eagerly load H5NeuroVec objects into memory
#' @return An h5_backend S3 object
#' @export
#' @keywords internal
h5_backend <- function(source, mask_source,
                       mask_dataset = "data/elements",
                       data_dataset = "data",
                       preload = FALSE) {
  # Check if fmristore is available FIRST
  if (!requireNamespace("fmristore", quietly = TRUE)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "Package 'fmristore' is required for H5 backend but is not available",
      parameter = "backend_type"
    )
  }

  # Validate inputs
  if (is.character(source)) {
    # File paths provided
    if (!all(file.exists(source))) {
      missing_files <- source[!file.exists(source)]
      stop_fmridataset(
        fmridataset_error_backend_io,
        message = sprintf("H5 source files not found: %s", paste(missing_files, collapse = ", ")),
        file = missing_files,
        operation = "open"
      )
    }
  } else if (is.list(source)) {
    # In-memory H5NeuroVec objects provided
    valid_types <- vapply(source, function(x) {
      inherits(x, "H5NeuroVec")
    }, logical(1))

    if (!all(valid_types)) {
      stop_fmridataset(
        fmridataset_error_config,
        message = "All source objects must be H5NeuroVec objects",
        parameter = "source"
      )
    }
  } else {
    stop_fmridataset(
      fmridataset_error_config,
      message = "source must be character vector (H5 file paths) or list (H5NeuroVec objects)",
      parameter = "source",
      value = class(source)
    )
  }

  # Validate mask source
  if (is.character(mask_source)) {
    if (!file.exists(mask_source)) {
      stop_fmridataset(
        fmridataset_error_backend_io,
        message = sprintf("H5 mask file not found: %s", mask_source),
        file = mask_source,
        operation = "open"
      )
    }
  } else if (!inherits(mask_source, "NeuroVol") && !inherits(mask_source, "H5NeuroVol")) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "mask_source must be file path, NeuroVol, or H5NeuroVol object",
      parameter = "mask_source",
      value = class(mask_source)
    )
  }

  backend <- new.env(parent = emptyenv())
  backend$source <- source
  backend$mask_source <- mask_source
  backend$mask_dataset <- mask_dataset
  backend$data_dataset <- data_dataset
  backend$preload <- preload
  backend$h5_objects <- NULL
  backend$mask <- NULL
  backend$mask_vec <- NULL
  backend$dims <- NULL
  backend$metadata <- NULL

  class(backend) <- c("h5_backend", "storage_backend")
  backend
}

#' @export
backend_open.h5_backend <- function(backend) {
  if (backend$preload && is.null(backend$h5_objects)) {
    # Load H5NeuroVec objects
    backend$h5_objects <- if (is.character(backend$source)) {
      # Load from H5 files
      tryCatch(
        {
          lapply(backend$source, function(file_path) {
            fmristore::H5NeuroVec(file_path, dataset_name = backend$data_dataset)
          })
        },
        error = function(e) {
          stop_fmridataset(
            fmridataset_error_backend_io,
            message = sprintf("Failed to load H5NeuroVec from files: %s", e$message),
            file = backend$source,
            operation = "read"
          )
        }
      )
    } else {
      # Use pre-loaded H5NeuroVec objects
      backend$source
    }

    # Load mask
    backend$mask <- if (is.character(backend$mask_source)) {
      tryCatch(
        {
          # Try to load as H5NeuroVol first, then fall back to regular volume
          if (endsWith(tolower(backend$mask_source), ".h5")) {
            # Load as H5NeuroVol and extract array
            h5_mask <- fmristore::H5NeuroVol(backend$mask_source, dataset_name = backend$mask_dataset)
            mask_array <- as.array(h5_mask)
            close(h5_mask) # Close the H5 handle
            neuroim2::NeuroVol(mask_array, space = space(backend$h5_objects[[1]]))
          } else {
            # Load as regular volume file
            suppressWarnings(neuroim2::read_vol(backend$mask_source))
          }
        },
        error = function(e) {
          stop_fmridataset(
            fmridataset_error_backend_io,
            message = sprintf("Failed to read H5 mask: %s", e$message),
            file = backend$mask_source,
            operation = "read"
          )
        }
      )
    } else {
      # Use in-memory mask object
      backend$mask_source
    }

    # Extract dimensions from first H5NeuroVec
    if (length(backend$h5_objects) > 0) {
      first_obj <- backend$h5_objects[[1]]
      d <- dim(first_obj)

      # Calculate total time dimension across all H5 files
      total_time <- if (length(backend$h5_objects) > 1) {
        sum(sapply(backend$h5_objects, function(obj) dim(obj)[4]))
      } else {
        d[4]
      }

      backend$dims <- list(
        spatial = d[1:3],
        time = total_time
      )
    }
  }

  backend
}

#' @export
backend_close.h5_backend <- function(backend) {
  # Close any open H5NeuroVec objects
  if (!is.null(backend$h5_objects)) {
    lapply(backend$h5_objects, function(obj) {
      tryCatch(close(obj), error = function(e) invisible(NULL))
    })
  }
  invisible(NULL)
}

#' @export
backend_get_dims.h5_backend <- function(backend) {
  if (!is.null(backend$dims)) {
    return(backend$dims)
  }

  # Get dimensions without loading full data
  if (is.character(backend$source)) {
    # Read from first H5 file to get spatial dimensions
    tryCatch(
      {
        first_h5 <- fmristore::H5NeuroVec(backend$source[1], dataset_name = backend$data_dataset)
        d <- dim(first_h5)
        close(first_h5) # Close immediately after getting dimensions

        # Calculate total time dimension across all files
        total_time <- if (length(backend$source) > 1) {
          sum(sapply(backend$source, function(file_path) {
            h5_obj <- fmristore::H5NeuroVec(file_path, dataset_name = backend$data_dataset)
            time_dim <- dim(h5_obj)[4]
            close(h5_obj)
            time_dim
          }))
        } else {
          d[4]
        }

        backend$dims <- list(spatial = d[1:3], time = total_time)
        backend$dims
      },
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          message = sprintf("Failed to read H5 dimensions: %s", e$message),
          file = backend$source[1],
          operation = "read_header"
        )
      }
    )
  } else {
    # In-memory H5NeuroVec objects
    first_obj <- backend$source[[1]]
    d <- dim(first_obj)
    total_time <- if (length(backend$source) > 1) {
      sum(sapply(backend$source, function(obj) dim(obj)[4]))
    } else {
      d[4]
    }

    backend$dims <- list(spatial = d[1:3], time = total_time)
    backend$dims
  }
}

#' @export
backend_get_mask.h5_backend <- function(backend) {
  if (!is.null(backend$mask_vec)) {
    return(backend$mask_vec)
  }

  if (!is.null(backend$mask)) {
    mask_vol <- backend$mask
  } else if (is.character(backend$mask_source)) {
    mask_vol <- tryCatch(
      {
        if (endsWith(tolower(backend$mask_source), ".h5")) {
          # Load as H5NeuroVol
          h5_mask <- fmristore::H5NeuroVol(backend$mask_source, dataset_name = backend$mask_dataset)
          mask_array <- as.array(h5_mask)
          close(h5_mask) # Close the H5 handle

          # Get space information from first data file if available
          if (is.character(backend$source) && length(backend$source) > 0) {
            first_h5 <- fmristore::H5NeuroVec(backend$source[1], dataset_name = backend$data_dataset)
            space_info <- space(first_h5)
            close(first_h5)
            neuroim2::NeuroVol(mask_array, space = space_info)
          } else {
            # Create with minimal space info
            neuroim2::NeuroVol(mask_array)
          }
        } else {
          # Load as regular volume file
          suppressWarnings(neuroim2::read_vol(backend$mask_source))
        }
      },
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          message = sprintf("Failed to read H5 mask: %s", e$message),
          file = backend$mask_source,
          operation = "read"
        )
      }
    )
  } else {
    # In-memory mask
    mask_vol <- backend$mask_source
  }

  # Convert to logical vector
  mask_vec <- as.logical(as.vector(mask_vol))

  # Validate mask
  if (any(is.na(mask_vec))) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "H5 mask contains NA values",
      parameter = "mask"
    )
  }

  if (sum(mask_vec) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "H5 mask contains no TRUE values",
      parameter = "mask"
    )
  }

  backend$mask <- mask_vol
  backend$mask_vec <- mask_vec

  backend$mask_vec
}

#' @export
backend_get_data.h5_backend <- function(backend, rows = NULL, cols = NULL) {
  # Get or load H5NeuroVec objects
  h5_objects <- if (!is.null(backend$h5_objects)) {
    backend$h5_objects
  } else {
    # Load on demand
    if (is.character(backend$source)) {
      tryCatch(
        {
          lapply(backend$source, function(file_path) {
            fmristore::H5NeuroVec(file_path, dataset_name = backend$data_dataset)
          })
        },
        error = function(e) {
          stop_fmridataset(
            fmridataset_error_backend_io,
            message = sprintf("Failed to load H5NeuroVec from files: %s", e$message),
            file = backend$source,
            operation = "read"
          )
        }
      )
    } else {
      backend$source
    }
  }

  # Get mask information
  mask_vec <- backend_get_mask(backend)
  voxel_indices <- which(mask_vec)

  # Extract data matrix in timepoints × voxels format
  if (length(h5_objects) == 1) {
    # Single H5NeuroVec object
    h5_obj <- h5_objects[[1]]
    data_matrix <- neuroim2::series(h5_obj, voxel_indices)
  } else {
    # Multiple H5NeuroVec objects - concatenate along time dimension
    data_matrices <- lapply(h5_objects, function(h5_obj) {
      neuroim2::series(h5_obj, voxel_indices)
    })
    data_matrix <- do.call(rbind, data_matrices)
  }

  # Close H5 objects if we loaded them on demand
  if (is.null(backend$h5_objects)) {
    lapply(h5_objects, function(obj) {
      tryCatch(close(obj), error = function(e) invisible(NULL))
    })
  }

  # Apply subsetting if requested
  if (!is.null(rows)) {
    data_matrix <- data_matrix[rows, , drop = FALSE]
  }

  if (!is.null(cols)) {
    data_matrix <- data_matrix[, cols, drop = FALSE]
  }

  data_matrix
}

#' @export
backend_get_metadata.h5_backend <- function(backend) {
  # Get metadata from first H5NeuroVec object
  h5_obj <- if (!is.null(backend$h5_objects)) {
    backend$h5_objects[[1]]
  } else if (is.character(backend$source)) {
    # Load temporarily to get metadata
    first_h5 <- fmristore::H5NeuroVec(backend$source[1], dataset_name = backend$data_dataset)
    on.exit(close(first_h5))
    first_h5
  } else {
    backend$source[[1]]
  }

  # Extract neuroimaging metadata
  space_obj <- space(h5_obj)

  list(
    format = "h5",
    affine = trans(space_obj),
    voxel_dims = spacing(space_obj),
    origin = origin(space_obj),
    dimensions = dim(space_obj),
    data_files = if (is.character(backend$source)) backend$source else NULL,
    mask_file = if (is.character(backend$mask_source)) backend$mask_source else NULL
  )
}
