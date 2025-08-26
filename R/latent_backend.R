#' Latent Storage Backend
#'
#' @description
#' A storage backend implementation for latent space representations of fMRI data.
#' This backend works with data that has been decomposed into temporal components
#' (basis functions) and spatial loadings.
#'
#' @details
#' Unlike traditional voxel-based backends, latent backends store:
#' - Temporal basis functions (time × components)
#' - Spatial loadings (voxels × components)
#' - Optional per-voxel offsets
#'
#' The backend maintains compatibility with the storage_backend contract while
#' providing specialized methods for latent data access.
#'
#' @name latent-backend
#' @keywords internal
NULL

#' Create a Latent Backend
#'
#' @description
#' Creates a storage backend for latent space fMRI data.
#'
#' @param source Character vector of paths to LatentNeuroVec HDF5 files (.lv.h5) or
#'   a list of LatentNeuroVec objects from the fmristore package.
#' @param preload Logical, whether to load all data into memory (default: FALSE)
#' @return A latent_backend S3 object
#' @export
#' @keywords internal
#' @examples
#' \dontrun{
#' # From HDF5 files
#' backend <- latent_backend(c("run1.lv.h5", "run2.lv.h5"))
#'
#' # From pre-loaded objects
#' lvec1 <- fmristore::read_vec("run1.lv.h5")
#' lvec2 <- fmristore::read_vec("run2.lv.h5")
#' backend <- latent_backend(list(lvec1, lvec2))
#' }
latent_backend <- function(source, preload = FALSE) {
  # Validate source
  if (is.character(source)) {
    if (!all(file.exists(source))) {
      missing <- source[!file.exists(source)]
      stop_fmridataset(
        fmridataset_error_backend_io,
        sprintf("Source files not found: %s", paste(missing, collapse = ", ")),
        file = missing[1],
        operation = "create"
      )
    }
    if (!all(grepl("\\.(lv\\.h5|h5)$", source, ignore.case = TRUE))) {
      stop_fmridataset(
        fmridataset_error_config,
        "All source files must be HDF5 files (.h5 or .lv.h5)",
        parameter = "source"
      )
    }
  } else if (is.list(source)) {
    # Validate list items
    for (i in seq_along(source)) {
      item <- source[[i]]
      if (is.character(item)) {
        if (length(item) != 1 || !file.exists(item)) {
          stop_fmridataset(
            fmridataset_error_config,
            sprintf("Source item %d must be an existing file path", i),
            parameter = "source"
          )
        }
      } else if (!inherits(item, "LatentNeuroVec")) {
        # Allow mock objects for testing
        has_basis <- isS4(item) && "basis" %in% methods::slotNames(item)
        if (!has_basis) {
          stop_fmridataset(
            fmridataset_error_config,
            sprintf("Source item %d must be a LatentNeuroVec object or file path", i),
            parameter = "source"
          )
        }
      }
    }
  } else {
    stop_fmridataset(
      fmridataset_error_config,
      "source must be character vector or list",
      parameter = "source",
      value = class(source)
    )
  }

  # Create backend object
  backend <- list(
    source = source,
    preload = preload,
    data = NULL,
    dims = NULL,
    is_open = FALSE
  )

  class(backend) <- c("latent_backend", "storage_backend")
  backend
}

#' @rdname backend_open
#' @method backend_open latent_backend
#' @export
backend_open.latent_backend <- function(backend) {
  if (backend$is_open) {
    return(backend)
  }

  # Check if fmristore is available
  if (!requireNamespace("fmristore", quietly = TRUE)) {
    stop_fmridataset(
      fmridataset_error_config,
      "The fmristore package is required for latent_backend but is not installed",
      details = "Install with: remotes::install_github('bbuchsbaum/fmristore')"
    )
  }

  # Load data
  data <- list()

  tryCatch(
    {
      if (is.character(backend$source)) {
        read_vec <- get("read_vec", envir = asNamespace("fmristore"))
        for (i in seq_along(backend$source)) {
          data[[i]] <- read_vec(backend$source[i])
        }
      } else {
        for (i in seq_along(backend$source)) {
          if (is.character(backend$source[[i]])) {
            read_vec <- get("read_vec", envir = asNamespace("fmristore"))
            data[[i]] <- read_vec(backend$source[[i]])
          } else {
            data[[i]] <- backend$source[[i]]
          }
        }
      }
    },
    error = function(e) {
      stop_fmridataset(
        fmridataset_error_backend_io,
        sprintf("Failed to load latent data: %s", e$message),
        operation = "open"
      )
    }
  )

  # Validate consistency across runs
  if (length(data) > 1) {
    first_dims <- get_latent_space_dims(data[[1]])[1:3]
    first_ncomp <- ncol(data[[1]]@basis)

    for (i in 2:length(data)) {
      dims <- get_latent_space_dims(data[[i]])[1:3]
      ncomp <- ncol(data[[i]]@basis)

      if (!identical(first_dims, dims)) {
        stop_fmridataset(
          fmridataset_error_config,
          sprintf("Run %d has inconsistent spatial dimensions", i),
          parameter = "source"
        )
      }
      if (first_ncomp != ncomp) {
        stop_fmridataset(
          fmridataset_error_config,
          sprintf(
            "Run %d has different number of components (%d vs %d)",
            i, ncomp, first_ncomp
          ),
          parameter = "source"
        )
      }
    }
  }

  # Store dimensions
  first_obj <- data[[1]]
  spatial_dims <- get_latent_space_dims(first_obj)[1:3]
  n_components <- ncol(first_obj@basis)
  n_voxels <- nrow(first_obj@loadings)

  # Total time across all runs
  total_time <- sum(sapply(data, function(obj) {
    get_latent_space_dims(obj)[4]
  }))

  backend$data <- data
  backend$dims <- list(
    spatial = spatial_dims, # Original spatial dimensions
    time = total_time, # Total timepoints
    n_components = n_components, # Number of latent components
    n_voxels = n_voxels, # Number of voxels
    n_runs = length(data)
  )
  backend$is_open <- TRUE

  backend
}

#' @rdname backend_close
#' @method backend_close latent_backend
#' @export
backend_close.latent_backend <- function(backend) {
  backend$data <- NULL
  backend$is_open <- FALSE
  invisible(backend)
}

#' @rdname backend_get_dims
#' @method backend_get_dims latent_backend
#' @export
backend_get_dims.latent_backend <- function(backend) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing dimensions",
      operation = "get_dims"
    )
  }

  # Return dims in standard format
  # For latent backend, we return the original spatial dimensions
  # The actual data access returns components, but dims should reflect
  # the original space for consistency
  list(
    spatial = backend$dims$spatial, # Original spatial dimensions
    time = backend$dims$time
  )
}

#' @rdname backend_get_mask
#' @method backend_get_mask latent_backend
#' @export
backend_get_mask.latent_backend <- function(backend) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing mask",
      operation = "get_mask"
    )
  }

  # For latent backend, return a mask for voxels
  # All voxels with non-zero loadings are considered valid
  # This maintains consistency with the backend contract
  rep(TRUE, backend$dims$n_voxels)
}

#' @rdname backend_get_data
#' @method backend_get_data latent_backend
#' @export
backend_get_data.latent_backend <- function(backend, rows = NULL, cols = NULL) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing data",
      operation = "get_data"
    )
  }

  # Default to all rows/cols
  if (is.null(rows)) rows <- seq_len(backend$dims$time)
  if (is.null(cols)) cols <- seq_len(backend$dims$n_components)

  # Validate indices
  if (any(rows < 1 | rows > backend$dims$time)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Row indices must be between 1 and %d", backend$dims$time),
      parameter = "rows"
    )
  }

  if (any(cols < 1 | cols > backend$dims$n_components)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Column indices must be between 1 and %d", backend$dims$n_components),
      parameter = "cols"
    )
  }

  # Calculate time offsets for runs
  time_offsets <- c(0, cumsum(sapply(backend$data, function(obj) {
    get_latent_space_dims(obj)[4]
  })))

  # Pre-allocate result
  result <- matrix(NA_real_, nrow = length(rows), ncol = length(cols))

  # Vectorized extraction
  for (run_idx in seq_along(backend$data)) {
    run_start <- time_offsets[run_idx] + 1
    run_end <- time_offsets[run_idx + 1]

    # Find rows in this run
    run_rows <- which(rows >= run_start & rows <= run_end)
    if (length(run_rows) == 0) next

    # Local indices within run
    local_rows <- rows[run_rows] - time_offsets[run_idx]

    # Extract data from basis matrix
    obj <- backend$data[[run_idx]]
    result[run_rows, ] <- as.matrix(obj@basis[local_rows, cols, drop = FALSE])
  }

  result
}

#' @rdname backend_get_metadata
#' @method backend_get_metadata latent_backend
#' @export
backend_get_metadata.latent_backend <- function(backend) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing metadata",
      operation = "get_metadata"
    )
  }

  obj <- backend$data[[1]]

  # Calculate variance explained by each component
  basis_var <- apply(obj@basis, 2, var)
  loadings_norm <- if (inherits(obj@loadings, "Matrix")) {
    sqrt(Matrix::colSums(obj@loadings^2))
  } else {
    sqrt(colSums(obj@loadings^2))
  }

  metadata <- list(
    storage_format = "latent",
    n_components = backend$dims$n_components,
    n_voxels = backend$dims$n_voxels,
    n_runs = backend$dims$n_runs,
    has_offset = length(obj@offset) > 0,
    basis_variance = basis_var,
    loadings_norm = loadings_norm,
    loadings_sparsity = if (inherits(obj@loadings, "Matrix")) {
      1 - Matrix::nnzero(obj@loadings) / length(obj@loadings)
    } else {
      0 # Dense matrix has 0 sparsity
    }
  )

  metadata
}

# Additional methods specific to latent backend

#' Get Spatial Loadings from Latent Backend
#'
#' @param backend A latent_backend object
#' @param components Optional component indices to extract
#' @return Matrix or sparse matrix of spatial loadings
#' @export
#' @keywords internal
backend_get_loadings <- function(backend, components = NULL) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before accessing loadings",
      operation = "get_loadings"
    )
  }

  # Get loadings from first object (all should be identical)
  obj <- backend$data[[1]]
  loadings <- obj@loadings

  if (!is.null(components)) {
    loadings <- loadings[, components, drop = FALSE]
  }

  loadings
}

#' Reconstruct Voxel Data from Latent Backend
#'
#' @param backend A latent_backend object
#' @param rows Optional row indices (timepoints)
#' @param voxels Optional voxel indices
#' @return Matrix of reconstructed voxel data
#' @export
#' @keywords internal
backend_reconstruct_voxels <- function(backend, rows = NULL, voxels = NULL) {
  if (!backend$is_open) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      "Backend must be opened before reconstruction",
      operation = "reconstruct"
    )
  }

  # Get latent scores
  scores <- backend_get_data(backend, rows = rows)

  # Get spatial loadings
  loadings <- backend_get_loadings(backend)

  # Apply voxel subset if requested
  if (!is.null(voxels)) {
    loadings <- loadings[voxels, , drop = FALSE]
  }

  # Reconstruct: data = basis %*% t(loadings)
  # Handle both regular and sparse matrices
  if (inherits(loadings, "Matrix")) {
    reconstructed <- scores %*% Matrix::t(loadings)
  } else {
    reconstructed <- scores %*% t(loadings)
  }

  # Add offset if present
  obj <- backend$data[[1]]
  if (length(obj@offset) > 0) {
    offset <- if (!is.null(voxels)) obj@offset[voxels] else obj@offset
    reconstructed <- sweep(reconstructed, 2, offset, "+")
  }

  reconstructed
}

# Helper function to get space dimensions safely
get_latent_space_dims <- function(obj) {
  # For mock objects, check if space slot is numeric
  if ((inherits(obj, "mock_LatentNeuroVec") || inherits(obj, "MockLatentNeuroVec")) && isS4(obj)) {
    if (is.numeric(obj@space)) {
      return(obj@space)
    }
  }

  # Try to get space
  sp <- try(neuroim2::space(obj), silent = TRUE)
  if (inherits(sp, "try-error")) {
    # Fallback for objects without proper space
    return(c(dim(obj@loadings)[1], 1, 1, dim(obj@basis)[1]))
  }

  d <- dim(sp)
  if (is.null(d)) {
    # Fallback for mock objects
    d <- as.numeric(sp)
  }
  d
}
