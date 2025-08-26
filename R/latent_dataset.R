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
#' @param source Character vector of file paths to LatentNeuroVec HDF5 files (.lv.h5) or
#'   a list of LatentNeuroVec objects from the fmristore package.
#' @param TR The repetition time in seconds.
#' @param run_length Vector of integers indicating the number of scans in each run.
#' @param event_table Optional data.frame containing event onsets and experimental variables.
#' @param base_path Base directory for relative file paths.
#' @param censor Optional binary vector indicating which scans to remove.
#' @param preload Logical indicating whether to preload all data into memory.
#'
#' @return A `latent_dataset` object with class `c("latent_dataset", "fmri_dataset")`.
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
#' # From pre-loaded objects
#' lvec1 <- fmristore::read_vec("run1.lv.h5")
#' lvec2 <- fmristore::read_vec("run2.lv.h5")
#' dataset <- latent_dataset(
#'   source = list(lvec1, lvec2),
#'   TR = 2,
#'   run_length = c(100, 100)
#' )
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
  
  # Process source paths
  if (is.character(source)) {
    source <- ifelse(
      grepl("^(/|[A-Za-z]:)", source), # Check if absolute path
      source,
      file.path(base_path, source)
    )
  }
  
  # Create the backend
  backend <- latent_backend(source = source, preload = preload)
  
  # Open backend to validate
  backend <- backend_open(backend)
  
  # Get dimensions from backend
  dims <- backend$dims
  assertthat::assert_that(sum(run_length) == dims$time,
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
      backend = backend,
      sampling_frame = frame,
      event_table = suppressMessages(tibble::as_tibble(event_table, .name_repair = "check_unique")),
      censor = censor,
      n_runs = length(run_length)
    ),
    class = c("latent_dataset", "fmri_dataset", "list")
  )
  
  dataset
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
  backend <- x$backend
  if (!backend$is_open) {
    stop("Dataset backend is not open")
  }
  
  # Use backend_get_data which returns latent scores for latent_backend
  backend_get_data(backend, rows = rows, cols = cols)
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
  backend <- x$backend
  if (!backend$is_open) {
    stop("Dataset backend is not open")
  }
  
  # Use backend-specific function for loadings
  backend_get_loadings(backend, components = components)
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
  backend <- x$backend
  if (!backend$is_open) {
    stop("Dataset backend is not open")
  }
  
  # Get metadata from backend which includes component info
  backend_get_metadata(backend)
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
  backend <- x$backend
  if (!backend$is_open) {
    stop("Dataset backend is not open")
  }
  
  # Use backend-specific reconstruction
  backend_reconstruct_voxels(backend, rows = rows, voxels = voxels)
}

#' @export
print.latent_dataset <- function(x, ...) {
  backend <- x$backend
  if (!backend$is_open) {
    backend <- backend_open(backend)
  }
  
  dims <- backend$dims
  metadata <- backend_get_metadata(backend)
  
  cat("Latent Dataset\n")
  cat("--------------\n")
  cat("Runs:", dims$n_runs, "\n")
  cat("Total timepoints:", dims$time, "\n")
  cat("Components:", dims$n_components, "\n")
  cat("Original voxels:", metadata$n_voxels, "\n")
  cat("TR:", x$sampling_frame$TR, "seconds\n")
  
  if (metadata$loadings_sparsity > 0) {
    cat("Loadings sparsity:", sprintf("%.1f%%", metadata$loadings_sparsity * 100), "\n")
  }
  
  invisible(x)
}

# Implement required fmri_dataset generics

#' @export
get_data.latent_dataset <- function(x, ...) {
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
  backend <- x$backend
  if (!backend$is_open) {
    backend <- backend_open(backend)
  }
  
  # For latent datasets, return a component mask (all TRUE)
  rep(TRUE, backend$dims$n_components)
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
  backend <- x$backend
  if (!backend$is_open) {
    backend <- backend_open(backend)
  }
  backend$dims$time
}