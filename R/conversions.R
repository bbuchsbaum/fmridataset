#' @importFrom neuroim2 series

# ========================================================================
# Type Conversion Methods for fMRI Datasets
# ========================================================================
#
# This file implements methods for the as.matrix_dataset() generic
# declared in all_generic.R. Provides conversion from various dataset
# types to matrix_dataset objects.
# ========================================================================

#' @export
as.matrix_dataset.matrix_dataset <- function(x, ...) {
  x  # Already a matrix_dataset
}

#' @export
as.matrix_dataset.fmri_mem_dataset <- function(x, ...) {
  # Get the data matrix
  bvec <- get_data(x)
  mask <- get_mask(x)
  datamat <- series(bvec, which(mask != 0))
  
  # Create matrix_dataset
  matrix_dataset(
    datamat = datamat,
    TR = x$sampling_frame$TR,
    run_length = x$sampling_frame$run_length,
    event_table = x$event_table
  )
}

#' @export
as.matrix_dataset.fmri_file_dataset <- function(x, ...) {
  # Get the data matrix
  vec <- get_data(x)
  mask <- get_mask(x)
  datamat <- series(vec, which(mask != 0))
  
  # Create matrix_dataset
  matrix_dataset(
    datamat = datamat,
    TR = x$sampling_frame$TR,
    run_length = x$sampling_frame$run_length,
    event_table = x$event_table
  )
} 