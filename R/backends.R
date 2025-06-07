#' Create a Latent Backend
#'
#' @description
#' Creates a storage backend for accessing latent space representations of fMRI data
#' using LatentNeuroVec objects from the fmristore package. This backend provides
#' efficient access to data stored in a compressed latent space format.
#'
#' @param source A character vector of file paths to LatentNeuroVec HDF5 files (.lv.h5),
#'   or a list of LatentNeuroVec objects from the fmristore package.
#' @param mask_source Optional mask source. If NULL, the mask will be extracted from
#'   the first LatentNeuroVec object.
#' @param preload Logical indicating whether to preload all data into memory.
#'   Default is FALSE (lazy loading).
#'
#' @return A `latent_backend` object that implements the storage backend interface.
#'
#' @details
#' The latent backend supports LatentNeuroVec objects which store fMRI data in a
#' compressed latent space representation using basis functions and spatial loadings.
#' This format is particularly efficient for data that can be well-represented by
#' a lower-dimensional basis (e.g., from PCA, ICA, or dictionary learning).
#'
#' **LatentNeuroVec Structure:**
#' - `basis`: Temporal components (n_timepoints × k_components)
#' - `loadings`: Spatial components (n_voxels × k_components)
#' - `offset`: Optional per-voxel offset terms
#' - Data is reconstructed as: `data = basis %*% t(loadings) + offset`
#'
#' **IMPORTANT: Data Access Behavior**
#' Unlike other backends that return voxel-wise data, the latent_backend returns
#' **latent scores** (the basis/temporal components) rather than reconstructed voxel data.
#' This is because:
#' - Analyses are typically performed in the latent space for efficiency
#' - The latent scores capture the temporal dynamics in the compressed representation
#' - Reconstructing to ambient voxel space defeats the purpose of the compression
#' - `backend_get_data()` returns a matrix of size (time × components), not (time × voxels)
#' - `backend_get_mask()` returns a logical vector indicating active components, not spatial voxels
#'
#' **Supported Input Types:**
#' - File paths to `.lv.h5` files (LatentNeuroVec HDF5 format)
#' - Pre-loaded LatentNeuroVec objects
#' - Mixed lists of files and objects
#'
#' @examples
#' \dontrun{
#' # From LatentNeuroVec HDF5 files
#' backend <- latent_backend(
#'   source = c("run1.lv.h5", "run2.lv.h5", "run3.lv.h5")
#' )
#'
#' # From pre-loaded LatentNeuroVec objects
#' lvec1 <- fmristore::read_vec("run1.lv.h5")
#' lvec2 <- fmristore::read_vec("run2.lv.h5")
#' backend <- latent_backend(source = list(lvec1, lvec2))
#'
#' # Mixed sources
#' backend <- latent_backend(
#'   source = list(lvec1, "run2.lv.h5", "run3.lv.h5")
#' )
#' }
#'
#' @seealso
#' \code{\link{h5_backend}}, \code{\link{nifti_backend}}, \code{\link{matrix_backend}}
#'
#' @export
latent_backend <- function(source, mask_source = NULL, preload = FALSE) {
  assert_that(is.logical(preload))

  # Validate and process source
  if (is.character(source)) {
    # All file paths
    assert_that(all(file.exists(source)),
      msg = "All source files must exist"
    )
    assert_that(all(grepl("\\.(lv\\.h5|h5)$", source, ignore.case = TRUE)),
      msg = "All source files must be HDF5 files (.h5 or .lv.h5)"
    )
  } else if (is.list(source)) {
    # List of objects and/or file paths
    for (i in seq_along(source)) {
      item <- source[[i]]
      if (is.character(item)) {
        assert_that(length(item) == 1 && file.exists(item),
          msg = paste("Source item", i, "must be an existing file path")
        )
        assert_that(grepl("\\.(lv\\.h5|h5)$", item, ignore.case = TRUE),
          msg = paste("Source file", i, "must be an HDF5 file")
        )
      } else {
        assert_that(inherits(item, "LatentNeuroVec"),
          msg = paste("Source item", i, "must be a LatentNeuroVec object or file path")
        )
      }
    }
  } else {
    stop("source must be a character vector of file paths or a list of LatentNeuroVec objects/file paths")
  }

  # Create the backend object
  backend <- structure(
    list(
      source = source,
      mask_source = mask_source,
      preload = preload,
      data = if (preload) NULL else NULL, # Will be populated by backend_open
      is_open = FALSE
    ),
    class = c("latent_backend", "storage_backend")
  )

  backend
}

# Latent Backend Methods Implementation ====

#' @export
backend_open.latent_backend <- function(backend) {
  if (backend$is_open) {
    return(backend)
  }

  # Check if fmristore package is available
  if (!requireNamespace("fmristore", quietly = TRUE)) {
    stop("The fmristore package is required for latent_backend but is not installed")
  }

  # Load all LatentNeuroVec objects
  source_data <- list()

  if (is.character(backend$source)) {
    # All file paths
    for (i in seq_along(backend$source)) {
      path <- backend$source[i]
      source_data[[i]] <- fmristore::read_vec(path)
    }
  } else if (is.list(backend$source)) {
    # Mixed list
    for (i in seq_along(backend$source)) {
      item <- backend$source[[i]]
      if (is.character(item)) {
        source_data[[i]] <- fmristore::read_vec(item)
      } else {
        source_data[[i]] <- item # Already a LatentNeuroVec
      }
    }
  }

  # Validate all objects are LatentNeuroVec
  for (i in seq_along(source_data)) {
    if (!inherits(source_data[[i]], "LatentNeuroVec")) {
      stop(paste("Item", i, "is not a LatentNeuroVec object"))
    }
  }

  # Check consistency across objects
  if (length(source_data) > 1) {
    first_obj <- source_data[[1]]
    first_space_dims <- dim(neuroim2::space(first_obj))[1:3]
    first_mask <- as.array(neuroim2::mask(first_obj))

    for (i in 2:length(source_data)) {
      obj <- source_data[[i]]
      space_dims <- dim(neuroim2::space(obj))[1:3]
      mask_array <- as.array(neuroim2::mask(obj))

      if (!identical(first_space_dims, space_dims)) {
        stop(paste("LatentNeuroVec", i, "has inconsistent spatial dimensions"))
      }

      if (!identical(first_mask, mask_array)) {
        stop(paste("LatentNeuroVec", i, "has inconsistent mask"))
      }
    }
  }

  # Store the loaded data
  backend$data <- source_data
  backend$is_open <- TRUE
  backend
}

#' @export
backend_close.latent_backend <- function(backend) {
  if (!backend$is_open) {
    return(backend)
  }

  # Close any HDF5 file handles if they exist
  if (!is.null(backend$data)) {
    for (obj in backend$data) {
      if (inherits(obj, "LatentNeuroVec") && !is.null(obj@map)) {
        # LatentNeuroVec objects may have HDF5 handles, but they're usually managed automatically
        # No explicit close needed for LatentNeuroVec in fmristore
      }
    }
  }

  backend$data <- NULL
  backend$is_open <- FALSE
  backend
}

#' @export
backend_get_dims.latent_backend <- function(backend) {
  if (!backend$is_open) {
    stop("Backend must be opened before getting dimensions")
  }

  if (length(backend$data) == 0) {
    stop("No data available in backend")
  }

  # Get dimensions from first object
  first_obj <- backend$data[[1]]
  space_dims <- dim(neuroim2::space(first_obj))

  # Calculate total time across all objects
  total_time <- sum(sapply(backend$data, function(obj) dim(neuroim2::space(obj))[4]))

  # For latent backends, the "space" dimensions refer to the original spatial dimensions
  # but the data dimensions are time x components
  n_components <- ncol(first_obj@basis)

  list(
    space = space_dims[1:3], # Original spatial dimensions (for reference)
    time = total_time, # Total time points across all runs
    n_runs = length(backend$data), # Number of runs
    n_components = n_components, # Number of latent components (actual data columns)
    data_dims = c(total_time, n_components) # Actual data matrix dimensions
  )
}

#' @export
backend_get_mask.latent_backend <- function(backend) {
  if (!backend$is_open) {
    stop("Backend must be opened before getting mask")
  }

  if (length(backend$data) == 0) {
    stop("No data available in backend")
  }

  # For latent backends, the "mask" represents which components are active
  # Since all components are typically used, we return a mask of all TRUE
  # This is different from spatial masks used in other backends
  first_obj <- backend$data[[1]]
  n_components <- ncol(first_obj@basis)

  # Return logical vector indicating all components are active
  rep(TRUE, n_components)
}

#' @export
backend_get_data.latent_backend <- function(backend, rows = NULL, cols = NULL) {
  if (!backend$is_open) {
    stop("Backend must be opened before getting data")
  }

  if (length(backend$data) == 0) {
    stop("No data available in backend")
  }

  # For latent backends, data consists of latent scores (basis functions)
  # NOT reconstructed voxel data. This is the key difference from other backends.

  # Get dimensions
  dims <- backend_get_dims(backend)
  first_obj <- backend$data[[1]]
  n_components <- ncol(first_obj@basis)

  # Default to all rows/cols if not specified
  if (is.null(rows)) {
    rows <- 1:dims$time
  }
  if (is.null(cols)) {
    cols <- 1:n_components
  }

  # Validate column indices (components, not voxels)
  if (any(cols < 1) || any(cols > n_components)) {
    stop(paste("Column indices must be between 1 and", n_components, "(number of components)"))
  }

  # Determine which runs contain the requested rows
  time_offsets <- c(0, cumsum(sapply(backend$data, function(obj) dim(neuroim2::space(obj))[4])))

  # Initialize result matrix (time x components)
  result <- matrix(0, nrow = length(rows), ncol = length(cols))

  for (row_idx in seq_along(rows)) {
    global_row <- rows[row_idx]

    # Find which run this row belongs to
    run_idx <- which(global_row > time_offsets & global_row <= time_offsets[-1])[1]

    if (is.na(run_idx)) {
      next # Skip invalid rows
    }

    # Calculate local row index within the run
    local_row <- global_row - time_offsets[run_idx]

    # Get the LatentNeuroVec object for this run
    obj <- backend$data[[run_idx]]

    # Extract latent scores (basis matrix) for this timepoint
    # The basis matrix is (time x components)
    latent_scores <- as.matrix(obj@basis[local_row, cols, drop = FALSE])

    # Store in result
    result[row_idx, ] <- latent_scores
  }

  result
}

#' @export
backend_get_metadata.latent_backend <- function(backend) {
  if (!backend$is_open) {
    stop("Backend must be opened before getting metadata")
  }

  if (length(backend$data) == 0) {
    stop("No data available in backend")
  }

  # Collect metadata from all LatentNeuroVec objects
  metadata <- list()

  for (i in seq_along(backend$data)) {
    obj <- backend$data[[i]]

    # Extract basic metadata
    space_obj <- neuroim2::space(obj)
    obj_dims <- dim(space_obj)

    run_meta <- list(
      run = i,
      n_timepoints = obj_dims[4],
      spatial_dims = obj_dims[1:3],
      spacing = neuroim2::spacing(space_obj),
      origin = neuroim2::origin(space_obj),
      n_components = ncol(obj@basis),
      label = if (length(obj@label) > 0) obj@label else paste("run", i),
      has_offset = length(obj@offset) > 0,
      basis_class = class(obj@basis)[1],
      loadings_class = class(obj@loadings)[1],
      loadings_sparsity = if (inherits(obj@loadings, "Matrix")) {
        Matrix::nnzero(obj@loadings) / length(obj@loadings)
      } else {
        1.0 # Dense matrix
      }
    )

    metadata[[i]] <- run_meta
  }

  names(metadata) <- paste0("run_", seq_along(metadata))
  metadata
}
