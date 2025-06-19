#' NIfTI Storage Backend
#'
#' @description
#' A storage backend implementation for NIfTI format neuroimaging data.
#' Supports both file-based and in-memory NIfTI data.
#'
#' @details
#' The NiftiBackend can work with:
#' - File paths to NIfTI images
#' - Pre-loaded niftiImage objects (from RNifti package)
#' - neuroim2 NeuroVec objects
#'
#' @name nifti-backend
#' @keywords internal
NULL

#' Create a NIfTI Backend
#'
#' @param source Character vector of file paths or list of in-memory niftiImage/NeuroVec objects
#' @param mask_source File path to mask or in-memory NeuroVol object
#' @param preload Logical, whether to eagerly load data into memory
#' @param mode Storage mode for file-backed data: 'normal', 'bigvec', 'mmap', or 'filebacked'
#' @return A nifti_backend S3 object
#' @export
#' @keywords internal
nifti_backend <- function(source, mask_source, preload = FALSE,
                          mode = c("normal", "bigvec", "mmap", "filebacked")) {
  mode <- match.arg(mode)

  # Validate inputs
  if (is.character(source)) {
    # File paths provided
    if (!all(file.exists(source))) {
      missing_files <- source[!file.exists(source)]
      stop_fmridataset(
        fmridataset_error_backend_io,
        message = sprintf("Source files not found: %s", paste(missing_files, collapse = ", ")),
        file = missing_files,
        operation = "open"
      )
    }
  } else if (is.list(source)) {
    # In-memory objects provided
    valid_types <- vapply(source, function(x) {
      inherits(x, "NeuroVec") || inherits(x, "niftiImage")
    }, logical(1))

    if (!all(valid_types)) {
      stop_fmridataset(
        fmridataset_error_config,
        message = "All source objects must be NeuroVec or niftiImage objects",
        parameter = "source"
      )
    }
  } else {
    stop_fmridataset(
      fmridataset_error_config,
      message = "source must be character vector (file paths) or list (in-memory objects)",
      parameter = "source",
      value = class(source)
    )
  }

  # Validate mask
  if (is.character(mask_source)) {
    if (!file.exists(mask_source)) {
      stop_fmridataset(
        fmridataset_error_backend_io,
        message = sprintf("Mask file not found: %s", mask_source),
        file = mask_source,
        operation = "open"
      )
    }
  } else if (!inherits(mask_source, "NeuroVol")) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "mask_source must be file path or NeuroVol object",
      parameter = "mask_source",
      value = class(mask_source)
    )
  }

  backend <- new.env(parent = emptyenv())
  backend$source <- source
  backend$mask_source <- mask_source
  backend$preload <- preload
  backend$mode <- mode
  backend$data <- NULL
  backend$mask <- NULL
  backend$mask_vec <- NULL
  backend$dims <- NULL
  backend$metadata <- NULL

  class(backend) <- c("nifti_backend", "storage_backend")
  backend
}

#' @export
backend_open.nifti_backend <- function(backend) {
  if (backend$preload && is.null(backend$data)) {
    # Load mask first
    backend$mask <- if (is.character(backend$mask_source)) {
      tryCatch(
        suppressWarnings(neuroim2::read_vol(backend$mask_source)),
        error = function(e) {
          stop_fmridataset(
            fmridataset_error_backend_io,
            message = sprintf("Failed to read mask: %s", e$message),
            file = backend$mask_source,
            operation = "read"
          )
        }
      )
    } else {
      backend$mask_source
    }

    # Load data
    backend$data <- if (is.character(backend$source)) {
      tryCatch(
        suppressWarnings(neuroim2::read_vec(backend$source, mask = backend$mask, mode = backend$mode)),
        error = function(e) {
          stop_fmridataset(
            fmridataset_error_backend_io,
            message = sprintf("Failed to read data: %s", e$message),
            file = backend$source,
            operation = "read"
          )
        }
      )
    } else {
      # Handle in-memory objects
      if (length(backend$source) > 1) {
        do.call(neuroim2::NeuroVecSeq, backend$source)
      } else {
        backend$source[[1]]
      }
    }

    # Extract dimensions
    if (inherits(backend$data, "NeuroVec")) {
      d <- dim(backend$data)
      backend$dims <- list(
        spatial = d[1:3],
        time = d[4]
      )
    }
  }

  backend
}

#' @export
backend_close.nifti_backend <- function(backend) {
  # For NIfTI backend, we don't need to explicitly close file handles
  # as neuroim2 manages this internally
  invisible(NULL)
}

#' @export
backend_get_dims.nifti_backend <- function(backend) {
  if (!is.null(backend$dims)) {
    return(backend$dims)
  }

  # Get dimensions without loading full data
  if (is.character(backend$source)) {
    # Read header only - neuroim2 doesn't support header_only, so read minimal data
    tryCatch(
      {
        vec <- suppressWarnings(neuroim2::read_vec(backend$source[1]))
        d <- dim(vec)
        # Sum time dimension across all files
        total_time <- if (length(backend$source) > 1) {
          sum(sapply(backend$source, function(f) {
            v <- suppressWarnings(neuroim2::read_vec(f))
            dim(v)[4]
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
          message = sprintf("Failed to read dimensions: %s", e$message),
          file = backend$source[1],
          operation = "read_header"
        )
      }
    )
  } else {
    # In-memory objects
    obj <- if (is.list(backend$source)) backend$source[[1]] else backend$source
    d <- dim(obj)
    total_time <- if (is.list(backend$source) && length(backend$source) > 1) {
      sum(sapply(backend$source, function(x) dim(x)[4]))
    } else {
      d[4]
    }

    backend$dims <- list(spatial = d[1:3], time = total_time)
    backend$dims
  }
}

#' @export
backend_get_mask.nifti_backend <- function(backend) {
  if (!is.null(backend$mask_vec)) {
    return(backend$mask_vec)
  }

  if (!is.null(backend$mask)) {
    mask_vol <- backend$mask
  } else if (is.character(backend$mask_source)) {
    mask_vol <- tryCatch(
      suppressWarnings(neuroim2::read_vol(backend$mask_source)),
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          message = sprintf("Failed to read mask: %s", e$message),
          file = backend$mask_source,
          operation = "read"
        )
      }
    )
  } else {
    mask_vol <- backend$mask_source
  }

  # Convert to logical vector
  mask_vec <- as.logical(as.vector(mask_vol))

  # Validate mask
  if (any(is.na(mask_vec))) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "Mask contains NA values",
      parameter = "mask"
    )
  }

  if (sum(mask_vec) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "Mask contains no TRUE values",
      parameter = "mask"
    )
  }

  backend$mask <- mask_vol
  backend$mask_vec <- mask_vec

  backend$mask_vec
}

#' @export
backend_get_data.nifti_backend <- function(backend, rows = NULL, cols = NULL) {
  # Get the full data first
  if (!is.null(backend$data)) {
    vec <- backend$data
  } else {
    # Load data on demand
    mask <- if (!is.null(backend$mask)) {
      backend$mask
    } else if (is.character(backend$mask_source)) {
      suppressWarnings(neuroim2::read_vol(backend$mask_source))
    } else {
      backend$mask_source
    }

    vec <- if (is.character(backend$source)) {
      tryCatch(
        suppressWarnings(neuroim2::read_vec(backend$source, mask = mask, mode = backend$mode)),
        error = function(e) {
          stop_fmridataset(
            fmridataset_error_backend_io,
            message = sprintf("Failed to read data: %s", e$message),
            file = backend$source,
            operation = "read"
          )
        }
      )
    } else {
      if (length(backend$source) > 1) {
        do.call(neuroim2::NeuroVecSeq, backend$source)
      } else {
        backend$source[[1]]
      }
    }
  }

  # Extract data matrix in timepoints × voxels format
  mask_vec <- backend_get_mask(backend)
  voxel_indices <- which(mask_vec)

  # Use neuroim2::series to extract time series data
  data_matrix <- neuroim2::series(vec, voxel_indices)

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
backend_get_metadata.nifti_backend <- function(backend) {
  if (!is.null(backend$metadata)) {
    return(backend$metadata)
  }

  # Extract metadata from first source
  if (is.character(backend$source)) {
    vec <- tryCatch(
      suppressWarnings(neuroim2::read_vec(backend$source[1])),
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          message = sprintf("Failed to read metadata: %s", e$message),
          file = backend$source[1],
          operation = "read_header"
        )
      }
    )
  } else {
    vec <- if (is.list(backend$source)) backend$source[[1]] else backend$source
  }

  # Extract key metadata
  metadata <- list(
    affine = neuroim2::trans(vec),
    voxel_dims = neuroim2::spacing(vec),
    space = neuroim2::space(vec),
    origin = neuroim2::origin(vec)
  )

  # Cache for future use
  backend$metadata <- metadata
  metadata
}
