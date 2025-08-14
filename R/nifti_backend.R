#' NIfTI Storage Backend
#'
#' @description
#' A storage backend implementation for NIfTI format neuroimaging data.
#' Supports both file-based and in-memory NIfTI data.
#'
#' @details
#' The NiftiBackend can work with:
#' - File paths to NIfTI images
#' - Pre-loaded neuroim2 NeuroVec objects
#'
#' @name nifti-backend
#' @importFrom neuroim2 read_header
#' @importFrom cachem cache_mem
#' @keywords internal
NULL

#' Create a NIfTI Backend
#'
#' @param source Character vector of file paths or list of in-memory NeuroVec objects
#' @param mask_source File path to mask or in-memory NeuroVol object
#' @param preload Logical, whether to eagerly load data into memory
#' @param mode Storage mode for file-backed data: 'normal', 'bigvec', 'mmap', or 'filebacked'
#' @param dummy_mode Logical, if TRUE allows non-existent files (for testing). Default FALSE.
#' @return A nifti_backend S3 object
#' @export
#' @keywords internal
nifti_backend <- function(source, mask_source, preload = FALSE,
                          mode = c("normal", "bigvec", "mmap", "filebacked"),
                          dummy_mode = FALSE) {
  mode <- match.arg(mode)

  # Validate inputs (skip file checks if dummy_mode is TRUE)
  if (is.character(source)) {
    # File paths provided
    if (!dummy_mode && !all(file.exists(source))) {
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
      inherits(x, "NeuroVec")
    }, logical(1))

    if (!all(valid_types)) {
      stop_fmridataset(
        fmridataset_error_config,
        message = "All source objects must be NeuroVec objects",
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

  # Validate mask (skip file check if dummy_mode is TRUE)
  if (is.character(mask_source)) {
    if (!dummy_mode && !file.exists(mask_source)) {
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
  backend$dummy_mode <- dummy_mode
  backend$data <- NULL
  backend$mask <- NULL
  backend$mask_vec <- NULL
  backend$dims <- NULL
  backend$metadata <- NULL
  backend$run_length <- NULL  # Store run_length if provided for dummy mode
  
  # Initialize backend-specific cache for metadata and masks
  backend$cache <- cachem::cache_mem(
    max_size = 64 * 1024^2,  # 64MB for backend metadata/masks
    evict = "lru"
  )

  class(backend) <- c("nifti_backend", "storage_backend")
  backend
}

#' @rdname backend_open
#' @method backend_open nifti_backend
#' @export
backend_open.nifti_backend <- function(backend) {
  # Handle dummy mode - set up placeholder dimensions
  if (isTRUE(backend$dummy_mode)) {
    # Set standard placeholder dimensions if not already set
    if (is.null(backend$dims)) {
      # Use run_length if provided, otherwise calculate from files
      if (!is.null(backend$run_length)) {
        total_time <- sum(backend$run_length)
      } else {
        # Calculate number of files/timepoints
        n_files <- if (is.character(backend$source)) {
          length(backend$source)
        } else if (is.list(backend$source)) {
          length(backend$source)
        } else {
          1
        }
        total_time <- 100L * n_files  # 100 timepoints per file
      }
      
      # Use standard fMRI dimensions
      backend$dims <- list(
        spatial = c(64L, 64L, 30L),  # Standard spatial dimensions
        time = total_time
      )
    }
    
    # Set placeholder mask (all voxels included by default)
    if (is.null(backend$mask_vec)) {
      backend$mask_vec <- rep(TRUE, prod(backend$dims$spatial))
    }
    
    return(backend)
  }
  
  # Normal mode - existing behavior
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

#' @rdname backend_close
#' @method backend_close nifti_backend
#' @export
backend_close.nifti_backend <- function(backend) {
  # For NIfTI backend, we don't need to explicitly close file handles
  # as neuroim2 manages this internally
  invisible(NULL)
}

#' @rdname backend_get_dims
#' @method backend_get_dims nifti_backend
#' @export
backend_get_dims.nifti_backend <- function(backend) {
  if (!is.null(backend$dims)) {
    return(backend$dims)
  }
  
  # Handle dummy mode
  if (isTRUE(backend$dummy_mode)) {
    # Use run_length if provided, otherwise calculate from files
    if (!is.null(backend$run_length)) {
      total_time <- sum(backend$run_length)
    } else {
      # Calculate number of files/timepoints  
      n_files <- if (is.character(backend$source)) {
        length(backend$source)
      } else if (is.list(backend$source)) {
        length(backend$source)
      } else {
        1
      }
      total_time <- 100L * n_files  # 100 timepoints per file
    }
    
    backend$dims <- list(
      spatial = c(64L, 64L, 30L),  # Standard spatial dimensions
      time = total_time
    )
    return(backend$dims)
  }

  # Get dimensions without loading full data
  if (is.character(backend$source)) {
    # Use read_header for efficient dimension extraction
    tryCatch(
      {
        # Read header from first file to get spatial dimensions
        header_info <- neuroim2::read_header(backend$source[1])
        header_dims <- if (inherits(header_info, "NIFTIMetaInfo")) {
          if (isS4(header_info)) {
            header_info@dims
          } else {
            # Handle mocked object
            header_info$dims
          }
        } else {
          dim(header_info)
        }
        spatial_dims <- as.integer(header_dims[1:3])
        
        # Sum time dimension across all files
        total_time <- if (length(backend$source) > 1) {
          sum(sapply(backend$source, function(f) {
            h <- neuroim2::read_header(f)
            if (inherits(h, "NIFTIMetaInfo")) {
              if (isS4(h)) {
                h@dims[4]
              } else {
                h$dims[4]
              }
            } else {
              dim(h)[4]
            }
          }))
        } else {
          header_dims[4]
        }

        backend$dims <- list(spatial = spatial_dims, time = as.integer(total_time))
        backend$dims
      },
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          message = sprintf("Failed to read dimensions from header: %s", e$message),
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

#' @rdname backend_get_mask
#' @method backend_get_mask nifti_backend
#' @export
backend_get_mask.nifti_backend <- function(backend) {
  if (!is.null(backend$mask_vec)) {
    return(backend$mask_vec)
  }
  
  # Handle dummy mode
  if (isTRUE(backend$dummy_mode)) {
    dims <- backend_get_dims(backend)
    backend$mask_vec <- rep(TRUE, prod(dims$spatial))
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

#' @rdname backend_get_data
#' @method backend_get_data nifti_backend
#' @export
backend_get_data.nifti_backend <- function(backend, rows = NULL, cols = NULL) {
  # Handle dummy mode - return zeros matrix
  if (isTRUE(backend$dummy_mode)) {
    dims <- backend_get_dims(backend)
    mask_vec <- backend_get_mask(backend)
    n_voxels <- sum(mask_vec)
    
    # Create empty data matrix
    data_matrix <- matrix(0, nrow = dims$time, ncol = n_voxels)
    
    # Apply subsetting if requested
    if (!is.null(rows)) {
      data_matrix <- data_matrix[rows, , drop = FALSE]
    }
    
    if (!is.null(cols)) {
      data_matrix <- data_matrix[, cols, drop = FALSE]
    }
    
    return(data_matrix)
  }
  
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

#' @rdname backend_get_metadata
#' @method backend_get_metadata nifti_backend
#' @export
backend_get_metadata.nifti_backend <- function(backend) {
  if (!is.null(backend$metadata)) {
    return(backend$metadata)
  }
  
  # Handle dummy mode - return placeholder metadata
  if (isTRUE(backend$dummy_mode)) {
    dims <- backend_get_dims(backend)
    
    # Create placeholder neurospace
    neurospace <- neuroim2::NeuroSpace(
      dim = dims$spatial,
      spacing = c(3, 3, 3),  # Standard 3mm isotropic voxels
      origin = c(0, 0, 0)
      # axes will be set to default by NeuroSpace
    )
    
    backend$metadata <- list(
      affine = neuroim2::trans(neurospace),
      voxel_dims = c(3, 3, 3),
      space = neurospace,
      origin = c(0, 0, 0),
      dims = c(dims$spatial, dims$time)
    )
    
    return(backend$metadata)
  }

  # Extract metadata from first source
  if (is.character(backend$source)) {
    # Use read_header for efficient metadata extraction
    header_info <- tryCatch(
      neuroim2::read_header(backend$source[1]),
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_backend_io,
          message = sprintf("Failed to read metadata from header: %s", e$message),
          file = backend$source[1],
          operation = "read_header"
        )
      }
    )
    
    # Extract key metadata from header
    # Note: header_info is a NIFTIMetaInfo object
    # We need to construct a NeuroSpace object to get the transformation matrix
    neurospace <- neuroim2::NeuroSpace(
      dim = if (isS4(header_info)) header_info@dims[1:3] else header_info$dims[1:3],
      spacing = if (isS4(header_info)) header_info@spacing[1:3] else header_info$spacing[1:3],
      origin = if (isS4(header_info)) header_info@origin else header_info$origin,
      axes = if (isS4(header_info)) header_info@spatial_axes else header_info$spatial_axes
    )
    
    metadata <- list(
      affine = neuroim2::trans(neurospace),
      voxel_dims = if (isS4(header_info)) header_info@spacing else header_info$spacing,
      space = neurospace,
      origin = if (isS4(header_info)) header_info@origin else header_info$origin,
      dims = if (inherits(header_info, "NIFTIMetaInfo")) {
        if (isS4(header_info)) {
          header_info@dims
        } else {
          header_info$dims
        }
      } else {
        dim(header_info)
      }  # Include full dimensions
    )
  } else {
    # In-memory objects
    vec <- if (is.list(backend$source)) backend$source[[1]] else backend$source
    
    # Extract key metadata from in-memory object
    metadata <- list(
      affine = neuroim2::trans(vec),
      voxel_dims = neuroim2::spacing(vec),
      space = neuroim2::space(vec),
      origin = neuroim2::origin(vec),
      dims = dim(vec)
    )
  }

  # Cache for future use
  backend$metadata <- metadata
  metadata
}
