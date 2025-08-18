#' @importFrom assertthat assert_that
#' @importFrom tibble as_tibble
#' @importFrom Matrix nnzero colSums
#' @importFrom methods slotNames
#' @importFrom neuroim2 space
#' @importFrom fmrihrf sampling_frame
NULL

#' Latent Dataset Interface
#'
#' @description
#' A specialized dataset interface for working with latent space representations
#' of fMRI data. Unlike traditional fMRI datasets that work with voxel-space data,
#' latent datasets operate on compressed representations using basis functions.
#'
#' This interface is designed for data that has been decomposed into temporal
#' components (basis functions) and spatial loadings, such as from PCA, ICA,
#' or dictionary learning methods.
#'
#' @details
#' ## Key Differences from Standard Datasets:
#' 
#' - **Data Access**: Returns latent scores (time × components) instead of voxel data
#' - **Mask**: Represents active components, not spatial voxels
#' - **Dimensions**: Component space rather than voxel space
#' - **Reconstruction**: Can optionally reconstruct to voxel space on demand
#'
#' ## Data Structure:
#' 
#' Latent representations store data as:
#' - `basis`: Temporal components (n_timepoints × k_components)
#' - `loadings`: Spatial components (n_voxels × k_components)  
#' - `offset`: Optional per-voxel offset terms
#' - Reconstruction: `data = basis %*% t(loadings) + offset`
#'
#' @name latent_dataset
#' @family latent_data
NULL

#' Create a Latent Dataset
#'
#' @description
#' Creates a dataset object for working with latent space representations of fMRI data.
#' This is the primary constructor for latent datasets.
#'
#' @param source Character vector of file paths to LatentNeuroVec HDF5 files (.lv.h5),
#'   a list of LatentNeuroVec objects from the fmristore package, or a single 
#'   LatentNeuroVec object (legacy interface).
#' @param TR The repetition time in seconds.
#' @param run_length Vector of integers indicating the number of scans in each run.
#' @param event_table Optional data.frame containing event onsets and experimental variables.
#' @param base_path Base directory for relative file paths (ignored for LatentNeuroVec objects).
#' @param censor Optional binary vector indicating which scans to remove (ignored for LatentNeuroVec objects).
#' @param preload Logical indicating whether to preload all data into memory (ignored for LatentNeuroVec objects).
#'
#' @return A `latent_dataset` object with class `c("latent_dataset", "fmri_dataset")` for the new interface,
#'   or `c("latent_dataset", "matrix_dataset", "fmri_dataset")` for the legacy interface.
#'
#' @export
#' @family latent_data
#'
#' @examples
#' \dontrun{
#' # From LatentNeuroVec files
#' dataset <- latent_dataset(
#'   source = c("run1.lv.h5", "run2.lv.h5"),
#'   TR = 2,
#'   run_length = c(100, 100)
#' )
#'
#' # Legacy interface with single LatentNeuroVec object
#' lvec <- fmristore::LatentNeuroVec(basis, loadings, space)
#' dataset <- latent_dataset(lvec, TR = 2, run_length = 100)
#'
#' # Access latent scores
#' scores <- get_latent_scores(dataset)
#' 
#' # Get component metadata
#' comp_info <- get_component_info(dataset)
#' }
latent_dataset <- function(source,
                          TR,
                          run_length,
                          event_table = data.frame(),
                          base_path = ".",
                          censor = NULL,
                          preload = FALSE) {
  
  # Support legacy parameter name 'lvec' via ... or direct call
  # Check if this is a LatentNeuroVec object (legacy interface)
  if (inherits(source, "LatentNeuroVec")) {
    # Legacy interface for single LatentNeuroVec object
    # Lazy check: make sure fmristore is installed (fmristore is not a hard dependency)
    if (!requireNamespace("fmristore", quietly = TRUE)) {
      stop("The 'fmristore' package is required to create a latent_dataset. Please install fmristore.",
        call. = FALSE
      )
    }

    # Ensure the total run length matches the number of time points in lvec
    lvec <- source  # Rename for clarity
    assertthat::assert_that(
      sum(run_length) == dim(lvec)[4],
      msg = "Sum of run lengths must equal the 4th dimension of lvec"
    )

    frame <- fmrihrf::sampling_frame(blocklens = run_length, TR = TR)

    ret <- list(
      lvec = lvec,
      datamat = lvec@basis,
      TR = TR,
      nruns = length(run_length),
      event_table = event_table,
      sampling_frame = frame,
      mask = rep(TRUE, ncol(lvec@basis))
    )

    class(ret) <- c("latent_dataset", "matrix_dataset", "fmri_dataset", "list")
    return(ret)
  }
  
  # Process source paths
  if (is.character(source)) {
    source <- ifelse(
      grepl("^(/|[A-Za-z]:)", source), # Check if absolute path
      source,
      file.path(base_path, source)
    )
  }
  
  # Create the underlying storage
  storage <- latent_storage(source = source, preload = preload)
  
  # Open storage to validate
  storage <- open_latent_storage(storage)
  
  # Validate dimensions
  dims <- get_latent_dims(storage)
  assert_that(sum(run_length) == dims$time,
    msg = sprintf(
      "Sum of run_length (%d) must equal total time points (%d)",
      sum(run_length), dims$time
    )
  )
  
  # Create sampling frame
  frame <- fmrihrf::sampling_frame(blocklens = run_length, TR = TR)
  
  # Handle censoring
  if (is.null(censor)) {
    censor <- rep(0, sum(run_length))
  }
  
  # Create the dataset object
  dataset <- structure(
    list(
      storage = storage,
      sampling_frame = frame,
      event_table = suppressMessages(tibble::as_tibble(event_table, .name_repair = "check_unique")),
      censor = censor,
      n_runs = length(run_length)
    ),
    class = c("latent_dataset", "fmri_dataset", "list")
  )
  
  dataset
}

#' Latent Storage Interface
#'
#' @description
#' Internal storage interface for latent data. This replaces the backend
#' interface for latent data to avoid LSP violations.
#'
#' @param source Source data (files or objects)
#' @param preload Whether to preload data
#' @return A latent_storage object
#' @keywords internal
latent_storage <- function(source, preload = FALSE) {
  # Validate source
  if (is.character(source)) {
    assert_that(all(file.exists(source)),
      msg = "All source files must exist"
    )
    assert_that(all(grepl("\\.(lv\\.h5|h5)$", source, ignore.case = TRUE)),
      msg = "All source files must be HDF5 files (.h5 or .lv.h5)"
    )
  } else if (is.list(source)) {
    # Validate list items
    for (i in seq_along(source)) {
      item <- source[[i]]
      if (is.character(item)) {
        assert_that(length(item) == 1 && file.exists(item),
          msg = paste("Source item", i, "must be an existing file path")
        )
      } else {
        # Check for required structure
        has_basis <- isS4(item) && "basis" %in% methods::slotNames(item)
        if (!inherits(item, "LatentNeuroVec") && 
            !inherits(item, "mock_LatentNeuroVec") && 
            !inherits(item, "MockLatentNeuroVec") && 
            !has_basis) {
          stop(paste("Source item", i, "must be a LatentNeuroVec object or file path"))
        }
      }
    }
  } else {
    stop("source must be character vector or list")
  }
  
  structure(
    list(
      source = source,
      preload = preload,
      data = NULL,
      is_open = FALSE
    ),
    class = "latent_storage"
  )
}

#' Open Latent Storage
#' @keywords internal
open_latent_storage <- function(storage) {
  if (storage$is_open) {
    return(storage)
  }
  
  # Check for fmristore
  if (!requireNamespace("fmristore", quietly = TRUE)) {
    stop("The fmristore package is required for latent datasets but is not installed")
  }
  
  # Load data
  data <- list()
  
  if (is.character(storage$source)) {
    read_vec <- get("read_vec", envir = asNamespace("fmristore"))
    for (i in seq_along(storage$source)) {
      data[[i]] <- read_vec(storage$source[i])
    }
  } else {
    for (i in seq_along(storage$source)) {
      if (is.character(storage$source[[i]])) {
        read_vec <- get("read_vec", envir = asNamespace("fmristore"))
        data[[i]] <- read_vec(storage$source[[i]])
      } else {
        data[[i]] <- storage$source[[i]]
      }
    }
  }
  
  # Validate consistency
  if (length(data) > 1) {
    # Check all have same spatial dimensions and components
    first_dims <- get_space_dims(data[[1]])[1:3]
    first_ncomp <- ncol(data[[1]]@basis)
    
    for (i in 2:length(data)) {
      dims <- get_space_dims(data[[i]])[1:3]
      ncomp <- ncol(data[[i]]@basis)
      
      if (!identical(first_dims, dims)) {
        stop(paste("Object", i, "has inconsistent spatial dimensions"))
      }
      if (first_ncomp != ncomp) {
        stop(paste("Object", i, "has different number of components"))
      }
    }
  }
  
  storage$data <- data
  storage$is_open <- TRUE
  storage
}

#' Get Latent Storage Dimensions
#' @keywords internal  
get_latent_dims <- function(storage) {
  if (!storage$is_open) {
    stop("Storage must be opened first")
  }
  
  first_obj <- storage$data[[1]]
  spatial_dims <- get_space_dims(first_obj)[1:3]
  n_components <- ncol(first_obj@basis)
  
  # Total time across all runs
  total_time <- sum(sapply(storage$data, function(obj) {
    get_space_dims(obj)[4]
  }))
  
  list(
    spatial = spatial_dims,      # Original spatial dimensions
    time = total_time,          # Total timepoints
    n_components = n_components, # Number of latent components
    n_runs = length(storage$data)
  )
}

# Helper to get space dimensions safely
get_space_dims <- function(obj) {
  # For mock objects, check if space slot is numeric
  if ((inherits(obj, "mock_LatentNeuroVec") || inherits(obj, "MockLatentNeuroVec")) && isS4(obj)) {
    if (is.numeric(obj@space)) {
      return(obj@space)
    }
  }
  
  sp <- neuroim2::space(obj)
  d <- dim(sp)
  if (is.null(d)) {
    # Fallback for mock objects
    d <- as.numeric(sp)
  }
  d
}

#' Get Latent Scores from Dataset
#'
#' @description
#' Extract the latent scores (temporal components) from a latent dataset.
#' This is the primary data access method for latent datasets.
#'
#' @param x A latent_dataset object
#' @param rows Optional row indices (timepoints) to extract
#' @param cols Optional column indices (components) to extract
#' @param ... Additional arguments
#'
#' @return Matrix of latent scores (time × components)
#' @export
#' @family latent_data
get_latent_scores <- function(x, rows = NULL, cols = NULL, ...) {
  UseMethod("get_latent_scores")
}

#' @export
get_latent_scores.latent_dataset <- function(x, rows = NULL, cols = NULL, ...) {
  storage <- x$storage
  if (!storage$is_open) {
    stop("Dataset storage is not open")
  }
  
  dims <- get_latent_dims(storage)
  
  # Default to all rows/cols
  if (is.null(rows)) {
    rows <- 1:dims$time
  }
  if (is.null(cols)) {
    cols <- 1:dims$n_components
  }
  
  # Validate indices
  if (any(cols < 1) || any(cols > dims$n_components)) {
    stop(paste("Column indices must be between 1 and", dims$n_components))
  }
  
  # Calculate time offsets for runs
  time_offsets <- c(0, cumsum(sapply(storage$data, function(obj) {
    get_space_dims(obj)[4]
  })))
  
  # Extract data
  result <- matrix(0, nrow = length(rows), ncol = length(cols))
  
  for (i in seq_along(rows)) {
    global_row <- rows[i]
    
    # Find which run contains this row
    run_idx <- which(global_row > time_offsets[-length(time_offsets)] & 
                     global_row <= time_offsets[-1])[1]
    
    if (is.na(run_idx)) next
    
    # Local row within run
    local_row <- global_row - time_offsets[run_idx]
    
    # Extract scores from basis matrix
    obj <- storage$data[[run_idx]]
    result[i, ] <- as.matrix(obj@basis[local_row, cols, drop = FALSE])
  }
  
  result
}

#' Get Spatial Loadings from Dataset
#'
#' @description
#' Extract the spatial loadings (spatial components) from a latent dataset.
#'
#' @param x A latent_dataset object
#' @param components Optional component indices to extract
#' @param ... Additional arguments
#'
#' @return Matrix or sparse matrix of spatial loadings (voxels × components)
#' @export
#' @family latent_data
get_spatial_loadings <- function(x, components = NULL, ...) {
  UseMethod("get_spatial_loadings")
}

#' @export
get_spatial_loadings.latent_dataset <- function(x, components = NULL, ...) {
  storage <- x$storage
  if (!storage$is_open) {
    stop("Dataset storage is not open")
  }
  
  # Get loadings from first object (all should be identical)
  obj <- storage$data[[1]]
  loadings <- obj@loadings
  
  if (!is.null(components)) {
    loadings <- loadings[, components, drop = FALSE]
  }
  
  loadings
}

#' Get Component Information
#'
#' @description
#' Get metadata about the latent components in the dataset.
#'
#' @param x A latent_dataset object
#' @param ... Additional arguments
#'
#' @return A list containing component metadata
#' @export
#' @family latent_data
get_component_info <- function(x, ...) {
  UseMethod("get_component_info")
}

#' @export
get_component_info.latent_dataset <- function(x, ...) {
  storage <- x$storage
  if (!storage$is_open) {
    stop("Dataset storage is not open")
  }
  
  dims <- get_latent_dims(storage)
  obj <- storage$data[[1]]
  
  # Calculate variance explained by each component
  basis_var <- apply(obj@basis, 2, var)
  loadings_norm <- if (inherits(obj@loadings, "Matrix")) {
    sqrt(Matrix::colSums(obj@loadings^2))
  } else {
    sqrt(colSums(obj@loadings^2))
  }
  
  list(
    n_components = dims$n_components,
    n_voxels = nrow(obj@loadings),
    basis_variance = basis_var,
    loadings_norm = loadings_norm,
    has_offset = length(obj@offset) > 0,
    loadings_sparsity = if (inherits(obj@loadings, "Matrix")) {
      1 - Matrix::nnzero(obj@loadings) / length(obj@loadings)
    } else {
      0  # Dense matrix has 0 sparsity
    }
  )
}

#' Reconstruct Voxel Data from Latent Representation
#'
#' @description
#' Reconstruct the full voxel-space data from the latent representation.
#' This is computationally expensive and should be used sparingly.
#'
#' @param x A latent_dataset object
#' @param rows Optional row indices (timepoints) to reconstruct
#' @param voxels Optional voxel indices to reconstruct
#' @param ... Additional arguments
#'
#' @return Matrix of reconstructed voxel data (time × voxels)
#' @export
#' @family latent_data
reconstruct_voxels <- function(x, rows = NULL, voxels = NULL, ...) {
  UseMethod("reconstruct_voxels")
}

#' @export
reconstruct_voxels.latent_dataset <- function(x, rows = NULL, voxels = NULL, ...) {
  # Get latent scores
  scores <- get_latent_scores(x, rows = rows)
  
  # Get spatial loadings
  loadings <- get_spatial_loadings(x)
  
  # Apply voxel subset if requested
  if (!is.null(voxels)) {
    loadings <- loadings[voxels, , drop = FALSE]
  }
  
  # Reconstruct: data = basis %*% t(loadings)
  reconstructed <- scores %*% t(loadings)
  
  # Add offset if present
  obj <- x$storage$data[[1]]
  if (length(obj@offset) > 0) {
    offset <- if (!is.null(voxels)) obj@offset[voxels] else obj@offset
    reconstructed <- sweep(reconstructed, 2, offset, "+")
  }
  
  reconstructed
}

#' @export
print.latent_dataset <- function(x, ...) {
  dims <- get_latent_dims(x$storage)
  comp_info <- get_component_info(x)
  
  cat("Latent Dataset\n")
  cat("--------------\n")
  cat("Runs:", dims$n_runs, "\n")
  cat("Total timepoints:", dims$time, "\n")
  cat("Components:", dims$n_components, "\n")
  cat("Original voxels:", comp_info$n_voxels, "\n")
  cat("TR:", x$sampling_frame$TR, "seconds\n")
  
  if (comp_info$loadings_sparsity > 0) {
    cat("Loadings sparsity:", sprintf("%.1f%%", comp_info$loadings_sparsity * 100), "\n")
  }
  
  invisible(x)
}

# Implement required fmri_dataset generics

#' @export
get_data.latent_dataset <- function(x, ...) {
  # Handle legacy interface
  if (!is.null(x$lvec)) {
    return(x$lvec@basis)
  }
  
  warning("get_data() on latent_dataset returns latent scores, not voxel data. ",
          "Use get_latent_scores() for clarity or reconstruct_voxels() for voxel data.")
  get_latent_scores(x, ...)
}

#' @export
get_data_matrix.latent_dataset <- function(x, ...) {
  get_latent_scores(x, ...)
}

#' @export
get_mask.latent_dataset <- function(x, ...) {
  # Handle legacy interface
  if (!is.null(x$mask)) {
    return(x$mask)
  }
  
  # For latent datasets, return a component mask (all TRUE)
  dims <- get_latent_dims(x$storage)
  rep(TRUE, dims$n_components)
}

#' @export
blocklens.latent_dataset <- function(x, ...) {
  x$sampling_frame$blocklens
}

#' @export
get_TR.latent_dataset <- function(x, ...) {
  tr <- x$sampling_frame$TR
  if (length(tr) > 1) {
    # Return the first TR value (they should all be the same)
    tr[1]
  } else {
    tr
  }
}

#' @export
n_runs.latent_dataset <- function(x, ...) {
  x$n_runs
}

#' @export
n_timepoints.latent_dataset <- function(x, ...) {
  get_latent_dims(x$storage)$time
}