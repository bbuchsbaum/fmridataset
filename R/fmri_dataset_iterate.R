#' Data Chunking and Iteration for fmri_dataset Objects
#'
#' This file implements the `data_chunks()` generic and methods for chunking
#' fMRI data for parallel processing, along with the `fmri_data_chunk` S3 class
#' and internal iterator mechanisms. Provides backwards compatibility with fmrireg
#' while supporting the new dataset types.
#'
#' @name fmri_dataset_iterate
NULL

# ============================================================================
# Ticket #16: Generic data_chunks method for fmri_dataset
# ============================================================================

#' Create Data Chunks for fmri_dataset Objects
#'
#' **Ticket #16**: Generic method for chunking fMRI data to enable parallel processing.
#' Creates an iterator that yields chunks of data based on voxels or timepoints.
#' Maintains backwards compatibility with existing fmrireg `data_chunks` interface.
#'
#' @param dataset An fmri_dataset object
#' @param nchunks Integer number of chunks to create (default: 1)
#' @param runwise Logical indicating whether to chunk by runs (default: FALSE)
#' @param by Character string indicating chunking dimension: "voxel", "timepoint", or "run" (default: "voxel")
#' @param apply_transformations Logical indicating whether to apply transformation pipeline (default: TRUE)
#' @param apply_preprocessing Logical indicating whether to apply preprocessing (alias for apply_transformations, default: TRUE)
#' @param verbose Logical indicating whether to print progress (default: FALSE)
#' @param ... Additional arguments passed to methods
#' 
#' @return An iterator object of class `c("fmri_chunk_iterator", "abstractiter", "iter")`
#'   that yields `fmri_data_chunk` objects when iterated
#' 
#' @details
#' **Chunking Strategies**:
#' \itemize{
#'   \item \strong{runwise=TRUE}: Creates one chunk per run
#'   \item \strong{by="voxel"}: Chunks across spatial dimension (voxels)
#'   \item \strong{by="timepoint"}: Chunks across temporal dimension (timepoints)
#' }
#' 
#' **Backwards Compatibility**: 
#' This implementation maintains compatibility with existing fmrireg `data_chunks()` usage:
#' - Same parameter names and behavior
#' - Returns compatible iterator with `nextElem()` method
#' - Chunks contain `data`, `voxel_ind`, `row_ind`, `chunk_num` fields
#' 
#' **Iterator Usage**:
#' ```r
#' # Use with foreach
#' iter <- data_chunks(dataset, nchunks = 4)
#' results <- foreach(chunk = iter) %dopar% {
#'   process_chunk(chunk$data)
#' }
#' 
#' # Manual iteration
#' iter <- data_chunks(dataset, runwise = TRUE)
#' while(TRUE) {
#'   tryCatch({
#'     chunk <- iter$nextElem()
#'     process_chunk(chunk)
#'   }, error = function(e) {
#'     if(grepl("StopIteration", e$message)) break
#'     stop(e)
#'   })
#' }
#' ```
#' 
#' @examples
#' \dontrun{
#' # Create dataset
#' dataset <- as.fmri_dataset(file_paths, TR = 2.0, run_lengths = c(200, 180))
#' 
#' # Chunk by voxels (default)
#' iter <- data_chunks(dataset, nchunks = 4)
#' 
#' # Chunk by runs
#' iter <- data_chunks(dataset, runwise = TRUE)
#' 
#' # Chunk by timepoints
#' iter <- data_chunks(dataset, nchunks = 10, by = "timepoint")
#' 
#' # Use with foreach for parallel processing
#' library(foreach)
#' results <- foreach(chunk = iter) %dopar% {
#'   colMeans(chunk$data)
#' }
#' }
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_data_chunk}}, \code{\link{get_data_matrix}}
data_chunks <- function(dataset, nchunks = 1, runwise = FALSE, by = c("voxel", "timepoint", "run"), 
                       apply_transformations = TRUE, apply_preprocessing = NULL, verbose = FALSE, ...) {
  if (!is.fmri_dataset(dataset)) {
    stop("dataset must be an fmri_dataset object")
  }
  
  by <- match.arg(by)
  
  # Handle parameter compatibility
  if (!is.null(apply_preprocessing)) {
    apply_transformations <- apply_preprocessing
  }
  
  # Handle legacy parameter compatibility
  if (by == "run") {
    runwise <- TRUE
  }
  
  if (runwise) {
    # Create runwise iterator (nchunks is ignored for compatibility with fmrireg)
    # In runwise mode, number of chunks always equals number of runs
    return(create_runwise_iterator(dataset, nchunks, apply_transformations, verbose, ...))
  } else if (by == "voxel") {
    # Create voxel-based iterator
    return(create_voxel_iterator(dataset, nchunks, apply_transformations, verbose, ...))
  } else if (by == "timepoint") {
    # Create timepoint-based iterator
    return(create_timepoint_iterator(dataset, nchunks, apply_transformations, verbose, ...))
  }
}

# ============================================================================
# Ticket #17: fmri_data_chunk S3 Class
# ============================================================================

#' Create fmri_data_chunk Object
#'
#' **Ticket #17**: Constructor for the `fmri_data_chunk` S3 class that represents
#' a chunk of fMRI data with associated metadata for pipeline processing.
#'
#' @param data Numeric matrix containing the chunk data (timepoints x voxels)
#' @param voxel_indices Integer vector of voxel indices in the original mask
#' @param timepoint_indices Integer vector of timepoint indices in the original time series
#' @param chunk_num Integer chunk number for identification
#' @param run_ids Integer vector indicating which runs are included in this chunk
#' @param total_chunks Integer total number of chunks (for compatibility)
#' @param metadata List containing additional metadata about the chunk
#' 
#' @return An `fmri_data_chunk` object
#' 
#' @details
#' The `fmri_data_chunk` class provides a standardized container for chunks of fMRI data
#' that can be processed in parallel pipelines. Each chunk contains:
#' \itemize{
#'   \item \strong{data}: The actual data matrix
#'   \item \strong{voxel_indices}: Spatial coordinates in original space
#'   \item \strong{timepoint_indices}: Temporal coordinates in original time series
#'   \item \strong{chunk_num}: Unique identifier for reconstruction
#'   \item \strong{run_ids}: Run membership information
#'   \item \strong{total_chunks}: Total number of chunks (for compatibility)
#'   \item \strong{metadata}: Additional context (TR, dimensions, etc.)
#' }
#' 
#' **Backwards Compatibility**: Maintains fmrireg field names `voxel_ind`, `row_ind` 
#' alongside new standardized names.
#' 
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{data_chunks}}, \code{\link{print.fmri_data_chunk}}
fmri_data_chunk <- function(data, 
                           voxel_indices, 
                           timepoint_indices = NULL, 
                           chunk_num,
                           run_ids = NULL,
                           total_chunks = NULL,
                           metadata = list()) {
  
  # Validate inputs
  if (!is.matrix(data) && !is.numeric(data)) {
    stop("data must be a numeric matrix")
  }
  
  if (is.vector(data)) {
    data <- matrix(data, ncol = 1)
  }
  
  if (length(voxel_indices) != ncol(data)) {
    stop("voxel_indices length must match number of columns in data")
  }
  
  # Provide default timepoint_indices if not specified
  if (is.null(timepoint_indices)) {
    timepoint_indices <- seq_len(nrow(data))
  }
  
  if (length(timepoint_indices) != nrow(data)) {
    stop("timepoint_indices length must match number of rows in data")
  }
  
  # Create chunk object
  chunk <- list(
    data = data,
    voxel_indices = as.integer(voxel_indices),
    timepoint_indices = as.integer(timepoint_indices),
    chunk_num = as.integer(chunk_num),
    run_ids = if (!is.null(run_ids)) as.integer(run_ids) else NULL,
    total_chunks = if (!is.null(total_chunks)) as.integer(total_chunks) else NULL,
    metadata = metadata,
    
    # Backwards compatibility with fmrireg field names
    voxel_ind = as.integer(voxel_indices),
    row_ind = as.integer(timepoint_indices)
  )
  
  class(chunk) <- c("fmri_data_chunk", "list")
  return(chunk)
}

#' Print Method for fmri_data_chunk
#'
#' **Ticket #17**: Print method for `fmri_data_chunk` objects that provides
#' a clean summary of chunk contents.
#'
#' @param x An `fmri_data_chunk` object
#' @param ... Additional arguments (ignored)
#' 
#' @export
#' @family fmri_dataset
print.fmri_data_chunk <- function(x, ...) {
  
  cat("\n═══ fmri_data_chunk ═══\n")
  
  # Basic info
  cat("\n📦 Chunk Information:\n")
  if (!is.null(x$total_chunks)) {
    cat("  • Chunk", x$chunk_num, "of", x$total_chunks, "\n")
  } else {
    cat("  • Chunk number:", x$chunk_num, "\n")
  }
  
  # Data dimensions
  cat("\n📊 Data Dimensions:\n")
  if (is.matrix(x$data)) {
    cat("  • Data matrix:", nrow(x$data), "×", ncol(x$data), "(timepoints × voxels)\n")
  } else {
    cat("  • Data vector: length", length(x$data), "\n")
  }
  
  # Spatial information
  cat("\n🧠 Spatial Information:\n")
  if (!is.null(x$voxel_indices)) {
    n_voxels <- length(x$voxel_indices)
    cat("  • Voxel indices:", n_voxels, "voxels\n")
    if (n_voxels > 0) {
      cat("    Range:", min(x$voxel_indices), "-", max(x$voxel_indices), "\n")
    }
  }
  
  # Temporal information
  cat("\n⏱️  Temporal Information:\n")
  if (!is.null(x$timepoint_indices)) {
    n_timepoints <- length(x$timepoint_indices)
    cat("  • Timepoint indices:", n_timepoints, "timepoints\n")
    if (n_timepoints > 0) {
      cat("    Range:", min(x$timepoint_indices), "-", max(x$timepoint_indices))
      if (n_timepoints <= 6) {
        cat(" (", paste(x$timepoint_indices, collapse = ", "), ")")
      } else {
        cat(" (", paste(x$timepoint_indices[1:3], collapse = ", "), ", ..., ", 
            paste(x$timepoint_indices[(n_timepoints-2):n_timepoints], collapse = ", "), ")")
      }
      cat("\n")
    }
  }
  
  # Run information
  if (!is.null(x$run_ids)) {
    cat("\n🏃 Run Information:\n")
    cat("  • Run IDs:", paste(unique(x$run_ids), collapse = ", "), "\n")
  }
  
  # Data summary
  if (is.matrix(x$data) || is.vector(x$data)) {
    cat("\n📈 Data Summary:\n")
    data_vec <- as.vector(x$data)
    if (length(data_vec) > 0) {
      cat("  • Values (sample): Min.:", round(min(data_vec, na.rm = TRUE), 3),
          ", Median:", round(median(data_vec, na.rm = TRUE), 3),
          ", Mean:", round(mean(data_vec, na.rm = TRUE), 3),
          ", Max.:", round(max(data_vec, na.rm = TRUE), 3), "\n")
    }
  }
  
  cat("\n")
  invisible(x)
}

#' Check if Object is fmri_data_chunk
#'
#' @param x Object to test
#' @return Logical indicating if x is an fmri_data_chunk
#' @export
#' @family fmri_dataset
is.fmri_data_chunk <- function(x) {
  inherits(x, "fmri_data_chunk")
}

# ============================================================================
# Ticket #18: Internal Iterator Mechanism
# ============================================================================

#' Create fMRI Chunk Iterator
#'
#' **Ticket #18**: Internal iterator mechanism that provides the backbone for
#' `data_chunks()` functionality. Creates iterators compatible with foreach
#' and manual iteration patterns.
#'
#' @param dataset An fmri_dataset object
#' @param nchunks Number of chunks
#' @param chunk_getter Function that retrieves a chunk given chunk number
#' @param strategy Character string describing chunking strategy for debugging
#' 
#' @return An iterator object with `nextElem()` method
#' @keywords internal
#' @noRd
fmri_chunk_iterator <- function(dataset, nchunks, chunk_getter, strategy = "unknown") {
  
  chunk_num <- 1
  
  # Iterator function that yields chunks
  nextElem <- function() {
    if (chunk_num > nchunks) {
      stop("StopIteration", call. = FALSE)
    }
    
    chunk <- chunk_getter(chunk_num)
    chunk_num <<- chunk_num + 1
    return(chunk)
  }
  
  # Reset function
  reset <- function() {
    chunk_num <<- 1
  }
  
  # Create iterator object
  iterator <- list(
    nchunks = nchunks,
    nextElem = nextElem,
    reset = reset,
    strategy = strategy,
    dataset_type = get_dataset_type(dataset)
  )
  
  class(iterator) <- c("fmri_chunk_iterator", "abstractiter", "iter")
  return(iterator)
}

#' Iterator method for fmri_chunk_iterator
#' 
#' Makes the iterator work with for loops by implementing the iter method
#' @param obj The fmri_chunk_iterator object
#' @return The iterator object itself
#' @keywords internal
#' @export
iter.fmri_chunk_iterator <- function(obj) {
  # Reset the iterator to start from the beginning
  reset_fn <- attr(obj, "reset", exact = TRUE)
  if (!is.null(reset_fn)) {
    reset_fn()
  }
  return(obj)
}

#' nextElem method for fmri_chunk_iterator
#' 
#' Extract method to call the nextElem function
#' @param obj The fmri_chunk_iterator object
#' @return The next chunk
#' @keywords internal  
#' @export
nextElem.fmri_chunk_iterator <- function(obj) {
  nextElem_fn <- attr(obj, "nextElem", exact = TRUE)
  if (!is.null(nextElem_fn)) {
    return(nextElem_fn())
  } else {
    stop("No nextElem function found in iterator")
  }
}

#' Custom $ method for fmri_chunk_iterator
#' 
#' Allows access to nextElem and reset functions via $ operator
#' @param x The fmri_chunk_iterator object
#' @param name The name of the attribute to access
#' @return The requested function or NULL
#' @keywords internal
#' @export
`$.fmri_chunk_iterator` <- function(x, name) {
  if (name == "nextElem") {
    return(attr(x, "nextElem", exact = TRUE))
  } else if (name == "reset") {
    return(attr(x, "reset", exact = TRUE))
  } else {
    # For other names, use default list behavior
    NextMethod("$")
  }
}

#' Print Method for fmri_chunk_iterator
#'
#' @param x An fmri_chunk_iterator object
#' @param ... Additional arguments (ignored)
#' @export
print.fmri_chunk_iterator <- function(x, ...) {
  cat("\n═══ fMRI Chunk Iterator ═══\n")
  cat("\n📋 Iterator Information:\n")
  cat("  • Total chunks:", attr(x, "nchunks", exact = TRUE), "\n")
  cat("  • Strategy:", attr(x, "strategy", exact = TRUE), "\n")
  cat("  • Dataset type:", attr(x, "dataset_type", exact = TRUE), "\n")
  
  cat("\n💡 Usage:\n")
  cat("  • foreach: foreach(chunk = iterator) %dopar% { ... }\n")
  cat("  • Manual: chunk <- iterator$nextElem()\n")
  cat("  • For loop: for(chunk in iterator) { ... }\n")
  
  cat("\n")
  invisible(x)
}

# ============================================================================
# Alternative Iterator Implementation for R for loops
# ============================================================================

#' Create List-based Iterator for fMRI chunks
#'
#' Creates an iterator that pre-computes all chunks and can work with R for loops
#'
#' @param dataset The fmri_dataset object
#' @param chunks_list List of pre-computed chunks
#' @param strategy Strategy description
#' @keywords internal
create_list_iterator <- function(dataset, chunks_list, strategy) {
  
  # Create an environment to hold iterator state
  iter_env <- new.env(parent = emptyenv())
  iter_env$chunks <- chunks_list
  iter_env$current <- 1
  iter_env$total <- length(chunks_list)
  
  # Iterator function
  nextElem <- function() {
    if (iter_env$current > iter_env$total) {
      stop("StopIteration", call. = FALSE)
    }
    
    chunk <- iter_env$chunks[[iter_env$current]]
    iter_env$current <- iter_env$current + 1
    return(chunk)
  }
  
  # Reset function
  reset <- function() {
    iter_env$current <- 1
  }
  
  # Create the iterator object as a special structure
  # Use the chunks_list as the base but add functions as attributes
  iterator <- structure(
    chunks_list,
    class = c("fmri_chunk_iterator", "list"),
    nextElem = nextElem,
    reset = reset,
    strategy = strategy,
    dataset_type = get_dataset_type(dataset),
    nchunks = length(chunks_list),
    total_chunks = length(chunks_list)
  )
  
  return(iterator)
}

#' Create Runwise Iterator (Updated)
#'
#' Creates iterator that chunks by runs (one chunk per run).
#'
#' @param dataset The fmri_dataset object
#' @param nchunks Number of chunks (ignored in runwise mode, provided for compatibility)
#' @param apply_transformations Whether to apply transformations
#' @param verbose Whether to print progress
#' @param ... Additional arguments for transformations
#' @keywords internal
create_runwise_iterator <- function(dataset, nchunks, apply_transformations, verbose, ...) {
  run_lengths <- get_run_lengths(dataset)
  n_runs <- length(run_lengths)
  
  # Note: nchunks is ignored in runwise mode - always creates one chunk per run
  # This maintains compatibility with fmrireg behavior
  
  # Pre-compute all chunks
  chunks_list <- list()
  
  for (current_run in 1:n_runs) {
    if (verbose) {
      cat("Creating chunk for run", current_run, "of", n_runs, "\n")
    }
    
    # Get data for this run
    run_data <- get_data_matrix(dataset, run_id = current_run, 
                               apply_transformations = apply_transformations,
                               verbose = verbose && current_run == 1, ...)
    
    # Get timepoint indices for this run
    timepoint_indices <- get_run_timepoint_indices(dataset, current_run)
    
    # Create chunk
    chunk <- fmri_data_chunk(
      data = run_data,
      voxel_indices = seq_len(ncol(run_data)),
      timepoint_indices = timepoint_indices,
      chunk_num = current_run,
      run_ids = current_run,
      total_chunks = n_runs
    )
    
    chunks_list[[current_run]] <- chunk
  }
  
  return(create_list_iterator(dataset, chunks_list, "runwise"))
}

#' Create Voxel Iterator (Updated)
#'
#' Creates iterator that chunks across spatial dimension (voxels).
#'
#' @param dataset The fmri_dataset object
#' @param nchunks Number of chunks to create
#' @param apply_transformations Whether to apply transformations
#' @param verbose Whether to print progress
#' @param ... Additional arguments for transformations
#' @keywords internal
create_voxel_iterator <- function(dataset, nchunks, apply_transformations, verbose, ...) {
  # Get total number of voxels
  n_voxels <- get_num_voxels(dataset)
  
  # Adjust nchunks if larger than n_voxels
  nchunks <- min(nchunks, n_voxels)
  
  # Handle edge case where nchunks might be 0
  if (nchunks < 1) {
    nchunks <- 1
  }
  
  # Calculate voxel assignments for each chunk
  if (nchunks == 1) {
    voxel_chunks <- list(seq_len(n_voxels))
  } else {
    voxel_chunks <- split(seq_len(n_voxels), cut(seq_len(n_voxels), nchunks, labels = FALSE))
  }
  
  # Load full data once (with transformations applied)
  full_data <- get_data_matrix(dataset, apply_transformations = apply_transformations,
                              verbose = verbose, ...)
  
  # Pre-compute all chunks
  chunks_list <- list()
  
  for (current_chunk in 1:length(voxel_chunks)) {
    if (verbose) {
      cat("Creating voxel chunk", current_chunk, "of", length(voxel_chunks), "\n")
    }
    
    # Get voxel indices for this chunk
    voxel_indices <- voxel_chunks[[current_chunk]]
    
    # Extract chunk data
    chunk_data <- full_data[, voxel_indices, drop = FALSE]
    
    # Create chunk
    chunk <- fmri_data_chunk(
      data = chunk_data,
      voxel_indices = voxel_indices,
      timepoint_indices = seq_len(nrow(chunk_data)),
      chunk_num = current_chunk,
      total_chunks = length(voxel_chunks)
    )
    
    chunks_list[[current_chunk]] <- chunk
  }
  
  return(create_list_iterator(dataset, chunks_list, "voxel"))
}

#' Create Timepoint Iterator (Updated)
#'
#' Creates iterator that chunks across temporal dimension (timepoints).
#'
#' @param dataset The fmri_dataset object  
#' @param nchunks Number of chunks to create
#' @param apply_transformations Whether to apply transformations
#' @param verbose Whether to print progress
#' @param ... Additional arguments for transformations
#' @keywords internal
create_timepoint_iterator <- function(dataset, nchunks, apply_transformations, verbose, ...) {
  # Load full data once (with transformations applied)
  full_data <- get_data_matrix(dataset, apply_transformations = apply_transformations,
                              verbose = verbose, ...)
  
  n_timepoints <- nrow(full_data)
  
  # Adjust nchunks if larger than n_timepoints
  nchunks <- min(nchunks, n_timepoints)
  
  # Handle edge case where nchunks might be 0
  if (nchunks < 1) {
    nchunks <- 1
  }
  
  # Calculate timepoint assignments for each chunk
  if (nchunks == 1) {
    timepoint_chunks <- list(seq_len(n_timepoints))
  } else {
    timepoint_chunks <- split(seq_len(n_timepoints), 
                             cut(seq_len(n_timepoints), nchunks, labels = FALSE))
  }
  
  # Pre-compute all chunks
  chunks_list <- list()
  
  for (current_chunk in 1:length(timepoint_chunks)) {
    if (verbose) {
      cat("Creating timepoint chunk", current_chunk, "of", length(timepoint_chunks), "\n")
    }
    
    # Get timepoint indices for this chunk
    timepoint_indices <- timepoint_chunks[[current_chunk]]
    
    # Extract chunk data
    chunk_data <- full_data[timepoint_indices, , drop = FALSE]
    
    # Create chunk
    chunk <- fmri_data_chunk(
      data = chunk_data,
      voxel_indices = seq_len(ncol(chunk_data)),
      timepoint_indices = timepoint_indices,
      chunk_num = current_chunk,
      total_chunks = length(timepoint_chunks)
    )
    
    chunks_list[[current_chunk]] <- chunk
  }
  
  return(create_list_iterator(dataset, chunks_list, "timepoint"))
}

# ============================================================================
# Helper Functions
# ============================================================================

#' Create Balanced Chunk Assignments
#'
#' Creates balanced assignments of indices to chunks.
#'
#' @param total_items Total number of items to chunk
#' @param nchunks Number of chunks to create
#' @return Integer vector of chunk assignments
#' @keywords internal
#' @noRd
create_balanced_chunks <- function(total_items, nchunks) {
  
  if (nchunks >= total_items) {
    # One item per chunk
    return(seq_len(total_items))
  }
  
  # Create balanced assignments
  chunk_size <- floor(total_items / nchunks)
  remainder <- total_items %% nchunks
  
  # Assign items to chunks
  assignments <- rep(seq_len(nchunks), each = chunk_size)
  
  # Distribute remainder
  if (remainder > 0) {
    extra_assignments <- seq_len(remainder)
    assignments <- c(assignments, extra_assignments)
  }
  
  return(assignments)
}

#' Backwards Compatibility Wrapper
#'
#' Creates a wrapper that maintains fmrireg compatibility for chunk objects.
#'
#' @param chunk An fmri_data_chunk object
#' @return Modified chunk with legacy field names
#' @keywords internal  
#' @noRd
create_legacy_chunk <- function(chunk) {
  # fmrireg expects these exact field names
  legacy_chunk <- list(
    data = chunk$data,
    voxel_ind = chunk$voxel_indices,
    row_ind = chunk$timepoint_indices,
    chunk_num = chunk$chunk_num
  )
  
  class(legacy_chunk) <- c("data_chunk", "list")
  return(legacy_chunk)
}

#' Get timepoint indices for a specific run
#' @param dataset The fmri_dataset object
#' @param run_id The run ID
#' @keywords internal
get_run_timepoint_indices <- function(dataset, run_id) {
  run_lengths <- get_run_lengths(dataset)
  
  if (run_id < 1 || run_id > length(run_lengths)) {
    stop("run_id must be between 1 and ", length(run_lengths))
  }
  
  # Calculate cumulative indices
  cum_lengths <- cumsum(c(0, run_lengths))
  start_idx <- cum_lengths[run_id] + 1
  end_idx <- cum_lengths[run_id + 1]
  
  start_idx:end_idx
} 