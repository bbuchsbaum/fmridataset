#' Create a Matrix Dataset Object (fmrireg compatibility)
#'
#' A convenience function that creates an fMRI dataset from a matrix of time-series data.
#' This function provides compatibility with the original fmrireg `matrix_dataset` interface
#' while using the modern fmridataset architecture under the hood.
#'
#' @param datamat A matrix where each column is a voxel time-series and each row is a timepoint
#' @param TR Repetition time (TR) of the fMRI acquisition in seconds
#' @param run_length A numeric vector specifying the length of each run in the dataset
#' @param event_table An optional data frame containing event information. Default is an empty data frame
#' @param base_path An optional base path for the dataset. Default is "." (current directory)
#' @param censor An optional logical or numeric vector specifying which time points to censor. Default is NULL
#' @param mask An optional logical vector specifying which voxels to include. Default is NULL (all voxels)
#'
#' @return An `fmri_dataset` object of class c("fmri_dataset", "list") that can be used
#'   with all modern fmridataset functions including `data_chunks`, `get_data_matrix`, etc.
#'
#' @details
#' This function provides a quick and convenient way to create an fMRI dataset from a matrix
#' of time-series data, similar to the original fmrireg `matrix_dataset` function. The main
#' differences from the original are:
#' \itemize{
#'   \item Returns a modern `fmri_dataset` object instead of `matrix_dataset`
#'   \item Uses `run_lengths` internally (vs `run_length`) for consistency
#'   \item Supports optional censoring and masking
#'   \item Integrates with the full fmridataset ecosystem
#' }
#'
#' **Matrix Format**: The input matrix should have timepoints as rows and voxels as columns.
#' This matches the expected format for most fMRI analysis workflows.
#'
#' **Run Structure**: The `run_length` parameter specifies how many timepoints belong to each
#' run. The sum of `run_length` must equal the number of rows in `datamat`.
#'
#' **Compatibility**: Objects created with this function work seamlessly with:
#' - `data_chunks()` for parallel processing
#' - `get_data_matrix()` for data access
#' - All transformation and preprocessing pipelines
#' - Legacy chunk tests from fmrireg
#'
#' @examples
#' # Basic usage with single run
#' X <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)  # 100 timepoints, 50 voxels
#' dset <- matrix_dataset(X, TR = 2, run_length = 100)
#'
#' # Multiple runs
#' Y <- matrix(rnorm(200 * 100), nrow = 200, ncol = 100)  # 200 timepoints, 100 voxels
#' dset <- matrix_dataset(Y, TR = 1.5, run_length = c(100, 100))
#'
#' # With event table
#' events <- data.frame(
#'   onset = c(10, 30, 60),
#'   duration = c(2, 2, 3),
#'   trial_type = c("A", "B", "A")
#' )
#' dset <- matrix_dataset(X, TR = 2, run_length = 100, event_table = events)
#'
#' # With censoring
#' censor_vec <- c(rep(FALSE, 95), rep(TRUE, 5))  # Censor last 5 timepoints
#' dset <- matrix_dataset(X, TR = 2, run_length = 100, censor = censor_vec)
#'
#' # With masking (only include first 30 voxels)
#' mask_vec <- c(rep(TRUE, 30), rep(FALSE, 20))
#' dset <- matrix_dataset(X, TR = 2, run_length = 100, mask = mask_vec)
#'
#' # Use with data_chunks (fmrireg style)
#' chunks <- data_chunks(dset, nchunks = 4)
#' chunk1 <- chunks$nextElem()
#'
#' # Use with foreach parallel processing
#' library(foreach)
#' results <- foreach(chunk = chunks) %do% {
#'   colMeans(chunk$data)
#' }
#'
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_dataset_create}}, \code{\link{data_chunks}}
matrix_dataset <- function(datamat, TR, run_length, event_table = data.frame(),
                          base_path = ".", censor = NULL, mask = NULL) {
  
  # Input validation
  if (is.vector(datamat)) {
    datamat <- as.matrix(datamat)
  }
  
  if (!is.matrix(datamat)) {
    stop("datamat must be a matrix or vector")
  }
  
  if (TR <= 0) {
    stop("TR must be positive")
  }
  
  if (any(run_length <= 0)) {
    stop("All run_length values must be positive")
  }
  
  if (sum(run_length) != nrow(datamat)) {
    stop("Sum of run_length (", sum(run_length), ") must equal number of rows in datamat (", nrow(datamat), ")")
  }
  
  # Handle censoring
  if (!is.null(censor)) {
    if (is.logical(censor)) {
      # Convert logical to indices (TRUE = censored)
      if (length(censor) != nrow(datamat)) {
        stop("censor vector length (", length(censor), ") must match number of timepoints (", nrow(datamat), ")")
      }
    } else if (is.numeric(censor)) {
      # Assume numeric censor is indices (1-based) of timepoints to censor
      if (any(censor < 1) || any(censor > nrow(datamat))) {
        stop("censor indices must be between 1 and ", nrow(datamat))
      }
      # Convert to logical
      censor_logical <- rep(FALSE, nrow(datamat))
      censor_logical[censor] <- TRUE
      censor <- censor_logical
    } else {
      stop("censor must be logical or numeric vector")
    }
  }
  
  # Handle masking
  if (!is.null(mask)) {
    if (is.logical(mask)) {
      if (length(mask) != ncol(datamat)) {
        stop("mask vector length (", length(mask), ") must match number of voxels (", ncol(datamat), ")")
      }
    } else {
      stop("mask must be a logical vector")
    }
  }
  
  # Create the dataset using fmri_dataset_create
  # Note: we use run_lengths (plural) for the modern interface
  dset <- fmri_dataset_create(
    images = datamat,
    TR = TR,
    run_lengths = run_length,  # Note: plural for consistency with modern interface
    event_table = if (nrow(event_table) > 0) event_table else NULL,
    base_path = base_path,
    censor = censor,
    mask = mask
  )
  
  # Add some metadata to indicate this came from matrix_dataset for compatibility
  dset$metadata$creation_method <- "matrix_dataset"
  dset$metadata$original_dims <- dim(datamat)
  
  # Add the matrix_dataset class to enable custom print method
  class(dset) <- c("matrix_dataset", class(dset))
  
  return(dset)
}

#' Check if Object is from matrix_dataset
#'
#' @param x Object to test
#' @return Logical indicating if x was created with matrix_dataset
#' @export
#' @family fmri_dataset
is.matrix_dataset <- function(x) {
  inherits(x, "matrix_dataset")
}

#' Print Method for matrix_dataset Objects
#'
#' @param x An fmri_dataset object created with matrix_dataset
#' @param ... Additional arguments (ignored)
#' @export
print.matrix_dataset <- function(x, ...) {
  cat("\n═══ Matrix Dataset (fmrireg compatible) ═══\n")
  
  # Basic info
  cat("\n📊 Matrix Information:\n")
  original_dims <- x$metadata$original_dims
  if (!is.null(original_dims)) {
    cat("  • Original matrix:", original_dims[1], "×", original_dims[2], "(timepoints × voxels)\n")
  }
  
  # Use the standard print method for the rest
  cat("\n")
  # Remove the matrix_dataset class temporarily to avoid infinite recursion
  class(x) <- class(x)[class(x) != "matrix_dataset"]
  print(x)
  invisible(x)
} 