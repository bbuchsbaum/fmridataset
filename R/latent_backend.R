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
#' Supports LatentNeuroVec objects from both the fmristore and fmrilatent packages.
#' fmristore is only required when source contains file paths (.lv.h5).
#' fmrilatent objects with lazy BasisHandle/LoadingsHandle slots are materialized
#' automatically on access.
#'
#' @name latent-backend
#' @keywords internal
NULL

# --- Internal helpers for slot materialization ---

#' Materialize a basis slot to a dense matrix
#'
#' Handles concrete matrices, sparse Matrix objects, and fmrilatent BasisHandle
#' objects. Returns a standard matrix.
#' @param obj A LatentNeuroVec object
#' @return A matrix (time x components)
#' @keywords internal
.materialize_basis <- function(obj) {
  b <- methods::slot(obj, "basis")
  if (is.matrix(b)) return(b)
  if (inherits(b, "Matrix")) return(as.matrix(b))
  # BasisHandle or other lazy type — try as.matrix dispatch
  tryCatch(
    as.matrix(b),
    error = function(e) {
      # Fallback: try fmrilatent accessor
      if (requireNamespace("fmrilatent", quietly = TRUE)) {
        return(as.matrix(fmrilatent::basis(obj)))
      }
      stop_fmridataset(
        fmridataset_error_backend_io,
        sprintf("Cannot materialize basis slot of class '%s': %s", class(b)[1], e$message),
        operation = "materialize_basis"
      )
    }
  )
}

#' Materialize a loadings slot to a matrix (possibly sparse)
#'
#' Handles concrete matrices, sparse Matrix objects, and fmrilatent LoadingsHandle
#' objects. Returns a matrix or sparse Matrix.
#' @param obj A LatentNeuroVec object
#' @return A matrix or Matrix (voxels x components)
#' @keywords internal
.materialize_loadings <- function(obj) {
  l <- methods::slot(obj, "loadings")
  if (is.matrix(l) || inherits(l, "Matrix")) return(l)
  # LoadingsHandle or other lazy type
  tryCatch(
    as.matrix(l),
    error = function(e) {
      if (requireNamespace("fmrilatent", quietly = TRUE)) {
        return(fmrilatent::loadings(obj))
      }
      stop_fmridataset(
        fmridataset_error_backend_io,
        sprintf("Cannot materialize loadings slot of class '%s': %s", class(l)[1], e$message),
        operation = "materialize_loadings"
      )
    }
  )
}

#' Get offset from a LatentNeuroVec object
#' @param obj A LatentNeuroVec object
#' @return Numeric vector (may be length 0)
#' @keywords internal
.get_offset <- function(obj) {
  methods::slot(obj, "offset")
}

#' Create a Latent Backend
#'
#' @description
#' Creates a storage backend for latent space fMRI data.
#'
#' @param source Character vector of paths to LatentNeuroVec HDF5 files (.lv.h5) or
#'   a list of LatentNeuroVec objects from the fmristore or fmrilatent packages.
#'   When file paths are provided, the fmristore package is required for reading.
#'   When in-memory LatentNeuroVec objects are provided, neither fmristore nor
#'   fmrilatent is required at runtime (though fmrilatent is used for lazy
#'   handle materialization if present).
#' @param preload Logical, whether to load all data into memory (default: FALSE)
#' @return A latent_backend S3 object
#' @export
#' @keywords internal
#' @examples
#' \dontrun{
#' # From HDF5 files (requires fmristore)
#' backend <- latent_backend(c("run1.lv.h5", "run2.lv.h5"))
#'
#' # From fmristore objects
#' lvec1 <- fmristore::read_vec("run1.lv.h5")
#' lvec2 <- fmristore::read_vec("run2.lv.h5")
#' backend <- latent_backend(list(lvec1, lvec2))
#'
#' # From fmrilatent objects (no fmristore needed)
#' lvec <- fmrilatent::encode(data_matrix, spec_time_dct(k = 15), mask = brain_mask)
#' backend <- latent_backend(list(lvec))
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

  # Create backend object using environment for reference semantics
  backend <- new.env(parent = emptyenv())
  backend$source <- source
  backend$preload <- preload
  backend$data <- NULL
  backend$dims <- NULL
  backend$is_open <- FALSE

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

  # Determine whether fmristore is needed (only for file-path sources)
  needs_fmristore <- is.character(backend$source) ||
    (is.list(backend$source) && any(vapply(backend$source, is.character, logical(1))))

  if (needs_fmristore && !requireNamespace("fmristore", quietly = TRUE)) {
    stop_fmridataset(
      fmridataset_error_config,
      "The fmristore package is required to read LatentNeuroVec files (.lv.h5)",
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
    first_ncomp <- ncol(.materialize_basis(data[[1]]))

    for (i in 2:length(data)) {
      dims <- get_latent_space_dims(data[[i]])[1:3]
      ncomp <- ncol(.materialize_basis(data[[i]]))

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
  n_components <- ncol(.materialize_basis(first_obj))
  n_voxels <- nrow(.materialize_loadings(first_obj))

  # Total time across all runs
  total_time <- sum(sapply(data, function(obj) {
    get_latent_space_dims(obj)[4]
  }))

  # Try to extract voxel indices for proper mask construction
  voxel_indices <- tryCatch(
    {
      idx <- neuroim2::indices(first_obj)
      if (is.numeric(idx) && length(idx) == n_voxels) idx else NULL
    },
    error = function(e) NULL
  )

  backend$data <- data
  backend$dims <- list(
    spatial = spatial_dims, # Original spatial dimensions
    time = total_time, # Total timepoints
    n_components = n_components, # Number of latent components
    n_voxels = n_voxels, # Number of voxels
    n_runs = length(data),
    voxel_indices = voxel_indices # Indices into full volume (may be NULL)
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
  invisible(NULL)
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

  # Return a full-volume mask consistent with the backend contract:
  # length(mask) == prod(spatial_dims)
  # Voxels with loadings are TRUE, all others FALSE.
  full_mask <- rep(FALSE, prod(backend$dims$spatial))
  if (!is.null(backend$dims$voxel_indices)) {
    full_mask[backend$dims$voxel_indices] <- TRUE
  } else {
    # Fallback: mark all voxels as valid (for mock/test objects)
    full_mask[] <- TRUE
  }
  full_mask
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

    # Extract data from basis matrix (handles both concrete and lazy types)
    obj <- backend$data[[run_idx]]
    basis_mat <- .materialize_basis(obj)
    result[run_rows, ] <- basis_mat[local_rows, cols, drop = FALSE]
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
  basis_mat <- .materialize_basis(obj)
  loadings_mat <- .materialize_loadings(obj)
  offset_vec <- .get_offset(obj)

  # Calculate variance explained by each component
  basis_var <- apply(basis_mat, 2, var)
  loadings_norm <- if (inherits(loadings_mat, "Matrix")) {
    sqrt(Matrix::colSums(loadings_mat^2))
  } else {
    sqrt(colSums(loadings_mat^2))
  }

  metadata <- list(
    storage_format = "latent",
    n_components = backend$dims$n_components,
    n_voxels = backend$dims$n_voxels,
    n_runs = backend$dims$n_runs,
    has_offset = length(offset_vec) > 0,
    basis_variance = basis_var,
    loadings_norm = loadings_norm,
    loadings_sparsity = if (inherits(loadings_mat, "Matrix")) {
      1 - Matrix::nnzero(loadings_mat) / length(loadings_mat)
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
  loadings <- .materialize_loadings(obj)

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
  offset_vec <- .get_offset(obj)
  if (length(offset_vec) > 0) {
    offset <- if (!is.null(voxels)) offset_vec[voxels] else offset_vec
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
