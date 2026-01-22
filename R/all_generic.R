# ========================================================================
# Generic Function Declarations for fmridataset Refactored Modules
# ========================================================================
#
# This file declares S3 generic functions for the refactored fMRI dataset
# functionality. These generics support the modular file structure and
# enable method dispatch across different dataset types.
#
# Note: This complements the existing aaa_generics.R which handles
# BIDS and other package-wide generics.
# ========================================================================

#' Generic Functions for fMRI Dataset Operations
#'
#' This file contains all generic function declarations for the refactored
#' fmridataset package. These establish the interface contracts that are
#' implemented by dataset-specific methods in other files.
#'
#' @name generics
NULL

#' Get Data from fMRI Dataset Objects
#'
#' Generic function to extract data from various fMRI dataset types.
#' Returns the underlying data in its native format (NeuroVec, matrix, etc.).
#'
#' @param x An fMRI dataset object (e.g., fmri_dataset, matrix_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' This function extracts the raw data from dataset objects, preserving
#' the original data type. For NeuroVec-based datasets, returns a NeuroVec
#' object. For matrix-based datasets, returns a matrix.
#'
#' @return Dataset-specific data object:
#'   \itemize{
#'     \item For \code{fmri_dataset}: Returns the underlying NeuroVec or matrix
#'     \item For \code{matrix_dataset}: Returns the data matrix
#'   }
#'
#' @examples
#' \donttest{
#' # Create a matrix dataset
#' mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
#' ds <- matrix_dataset(mat, TR = 2, run_length = 100)
#'
#' # Extract the data
#' data <- get_data(ds)
#' identical(data, mat) # TRUE
#' }
#'
#' @seealso
#' \code{\link{get_data_matrix}} for extracting data as a matrix,
#' \code{\link{get_mask}} for extracting the mask
#' @export
get_data <- function(x, ...) {
  UseMethod("get_data")
}

#' Get Data Matrix from fMRI Dataset Objects
#'
#' Generic function to extract data as a matrix from various fMRI dataset types.
#' Always returns a matrix with timepoints as rows and voxels as columns.
#'
#' @param x An fMRI dataset object (e.g., fmri_dataset, matrix_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' This function provides a unified interface for accessing fMRI data as a
#' matrix, regardless of the underlying storage format. The returned matrix
#' always has timepoints in rows and voxels in columns, matching the
#' conventional fMRI data organization.
#'
#' @return A numeric matrix with dimensions:
#'   \itemize{
#'     \item Rows: Number of timepoints
#'     \item Columns: Number of voxels (within mask)
#'   }
#'
#' @examples
#' \donttest{
#' # Create a matrix dataset
#' mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
#' ds <- matrix_dataset(mat, TR = 2, run_length = 100)
#'
#' # Extract as matrix
#' data_mat <- get_data_matrix(ds)
#' dim(data_mat) # 100 x 50
#' }
#'
#' @seealso
#' \code{\link{get_data}} for extracting data in native format,
#' \code{\link{as.matrix_dataset}} for converting to matrix dataset
#' @export
get_data_matrix <- function(x, ...) {
  UseMethod("get_data_matrix")
}

#' Get Mask from fMRI Dataset Objects
#'
#' Generic function to extract masks from various fMRI dataset types.
#' Returns the mask in its appropriate format for the dataset type.
#'
#' @param x An fMRI dataset object (e.g., fmri_dataset, matrix_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' The mask defines which voxels are included in the analysis. Different
#' dataset types may store masks in different formats (logical vectors,
#' NeuroVol objects, etc.). This function provides a unified interface
#' for mask extraction.
#'
#' @return Mask object appropriate for the dataset type:
#'   \itemize{
#'     \item For \code{matrix_dataset}: Logical vector
#'     \item For \code{fmri_dataset}: NeuroVol or logical vector
#'   }
#'
#' @examples
#' \donttest{
#' # Create a matrix dataset (matrix_dataset creates default mask internally)
#' mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
#' ds <- matrix_dataset(mat, TR = 2, run_length = 100)
#'
#' # Extract the mask (matrix_dataset creates all TRUE mask by default)
#' extracted_mask <- get_mask(ds)
#' sum(extracted_mask) # 50 (all TRUE values)
#' }
#'
#' @seealso
#' \code{\link{get_data}} for extracting data,
#' \code{\link{get_data_matrix}} for extracting data as matrix
#' @export
get_mask <- function(x, ...) {
  UseMethod("get_mask")
}

#' Get Block Lengths from Objects
#'
#' Generic function to extract block/run lengths from various objects.
#' Extends the sampling_frame generic to work with dataset objects.
#'
#' @param x An object with block structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' In fMRI experiments, data is often collected in multiple runs or blocks.
#' This function extracts the length (number of timepoints) of each run.
#' The sum of block lengths equals the total number of timepoints.
#'
#' @return Integer vector where each element represents the number of
#'   timepoints in the corresponding run/block
#'
#' @examples
#' \donttest{
#' # Create a dataset with 3 runs
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
#' blocklens(sf) # c(100, 120, 110)
#'
#' # Create dataset with multiple runs
#' mat <- matrix(rnorm(330 * 50), nrow = 330, ncol = 50)
#' ds <- matrix_dataset(mat, TR = 2, run_length = c(100, 120, 110))
#' blocklens(ds) # c(100, 120, 110)
#' }
#'
#' @seealso
#' \code{\link{n_runs}} for number of runs,
#' \code{\link{n_timepoints}} for total timepoints,
#' \code{\link{get_run_lengths}} for alternative function name
#' @export
blocklens <- function(x, ...) {
  UseMethod("blocklens")
}

#' Create Data Chunks for Processing
#'
#' Generic function to create data chunks for parallel processing from
#' various fMRI dataset types. Supports different chunking strategies.
#'
#' @param x An fMRI dataset object
#' @param nchunks Number of chunks to create (default: 1)
#' @param runwise If TRUE, create run-wise chunks (default: FALSE)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' Large fMRI datasets can be processed more efficiently by dividing them
#' into chunks. This function creates an iterator that yields data chunks
#' for parallel or sequential processing. Two chunking strategies are supported:
#' \itemize{
#'   \item Equal-sized chunks: Divides voxels into approximately equal groups
#'   \item Run-wise chunks: Each chunk contains all voxels from one or more complete runs
#' }
#'
#' @return A chunk iterator object that yields data chunks when iterated
#'
#' @examples
#' \donttest{
#' # Create a dataset
#' mat <- matrix(rnorm(100 * 1000), nrow = 100, ncol = 1000)
#' ds <- matrix_dataset(mat, TR = 2, run_length = 100)
#'
#' # Create 4 chunks for parallel processing
#' chunks <- data_chunks(ds, nchunks = 4)
#'
#' # Process chunks (example)
#' # results <- foreach(chunk = chunks) %dopar% {
#' #   process_chunk(chunk)
#' # }
#' }
#'
#' @seealso
#' \code{\link[iterators]{iter}} for iteration concepts
#' @export
data_chunks <- function(x, nchunks = 1, runwise = FALSE, ...) {
  UseMethod("data_chunks")
}

#' Convert to Matrix Dataset
#'
#' Generic function to convert various fMRI dataset types to matrix_dataset objects.
#' Provides a unified interface for getting matrix-based representations.
#'
#' @param x An fMRI dataset object
#' @param ... Additional arguments passed to methods
#'
#' @details
#' This function converts different dataset representations to the standard
#' matrix_dataset format, which stores data as a matrix with timepoints in
#' rows and voxels in columns. This is useful for algorithms that require
#' matrix operations or when a consistent data format is needed.
#'
#' @return A matrix_dataset object with the same data as the input
#'
#' @examples
#' \donttest{
#' # Convert various dataset types to matrix_dataset
#' # (example requires actual dataset object)
#' # mat_ds <- as.matrix_dataset(some_dataset)
#' }
#'
#' @seealso
#' \code{\link{matrix_dataset}} for creating matrix datasets,
#' \code{\link{get_data_matrix}} for extracting data as matrix
#' @export
as.matrix_dataset <- function(x, ...) {
  UseMethod("as.matrix_dataset")
}

# Sampling frame generics
#' Get TR (Repetition Time) from Sampling Frame
#'
#' Extracts the repetition time (TR) in seconds from objects containing
#' temporal information about fMRI acquisitions.
#'
#' @param x An object containing temporal information (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' The TR (repetition time) is the time between successive acquisitions
#' of the same slice in an fMRI scan, typically measured in seconds.
#' This parameter is crucial for temporal analyses and hemodynamic modeling.
#'
#' @return Numeric value representing TR in seconds
#'
#' @examples
#' \donttest{
#' # Create a sampling frame with TR = 2 seconds
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
#' get_TR(sf) # Returns: 2
#' }
#'
#' @seealso
#' [fmrihrf::sampling_frame()] for creating temporal structures,
#' \code{\link{get_total_duration}} for total scan duration
#' @export
get_TR <- function(x, ...) {
  UseMethod("get_TR")
}

#' Get Run Lengths from Sampling Frame
#'
#' Extracts the lengths of individual runs/blocks from objects containing
#' temporal structure information.
#'
#' @param x An object containing temporal structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' This function is synonymous with \code{\link{blocklens}} but uses
#' terminology more common in fMRI analysis. Each run represents a
#' continuous acquisition period, and the run length is the number
#' of timepoints (volumes) in that run.
#'
#' @return Integer vector where each element represents the number of
#'   timepoints in the corresponding run
#'
#' @examples
#' \donttest{
#' # Create a sampling frame with 3 runs
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
#' get_run_lengths(sf) # Returns: c(100, 120, 110)
#' }
#'
#' @seealso
#' \code{\link{blocklens}} for equivalent function,
#' \code{\link{n_runs}} for number of runs,
#' \code{\link{n_timepoints}} for total timepoints
#' @export
get_run_lengths <- function(x, ...) {
  UseMethod("get_run_lengths")
}

#' Get Number of Runs from Sampling Frame
#'
#' Extracts the total number of runs/blocks from objects containing
#' temporal structure information.
#'
#' @param x An object containing temporal structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @return Integer representing the total number of runs
#'
#' @examples
#' \donttest{
#' # Create a sampling frame with 3 runs
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
#' n_runs(sf) # Returns: 3
#' }
#'
#' @seealso
#' \code{\link{get_run_lengths}} for individual run lengths,
#' \code{\link{n_timepoints}} for total timepoints
#' @export
n_runs <- function(x, ...) {
  UseMethod("n_runs")
}

#' Get Number of Timepoints from Sampling Frame
#'
#' Extracts the total number of timepoints (volumes) across all runs
#' from objects containing temporal structure information.
#'
#' @param x An object containing temporal structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @return Integer representing the total number of timepoints
#'
#' @examples
#' \donttest{
#' # Create a sampling frame with 3 runs
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120, 110), TR = 2)
#' n_timepoints(sf) # Returns: 330 (sum of run lengths)
#' }
#'
#' @seealso
#' \code{\link{n_runs}} for number of runs,
#' \code{\link{get_run_lengths}} for individual run lengths
#' @export
n_timepoints <- function(x, ...) {
  UseMethod("n_timepoints")
}

#' Get Block IDs from Sampling Frame
#'
#' Generates a vector of block/run identifiers for each timepoint.
#'
#' @param x An object containing temporal structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' This function creates a vector where each element indicates which
#' run/block the corresponding timepoint belongs to. This is useful
#' for run-wise analyses or modeling run effects.
#'
#' @return Integer vector of length equal to total timepoints, with
#'   values indicating run membership (1 for first run, 2 for second, etc.)
#'
#' @examples
#' \donttest{
#' # Create a sampling frame with 2 runs of different lengths
#' sf <- fmrihrf::sampling_frame(blocklens = c(3, 4), TR = 2)
#' blockids(sf) # Returns: c(1, 1, 1, 2, 2, 2, 2)
#' }
#'
#' @seealso
#' \code{\link{get_run_lengths}} for run lengths,
#' \code{\link{samples}} for timepoint indices
#' @export
blockids <- function(x, ...) {
  UseMethod("blockids")
}

#' Get Sample Indices from Sampling Frame
#'
#' Generates a vector of timepoint indices, typically used for
#' time series analysis or indexing operations.
#'
#' @param x An object containing temporal structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @return Integer vector from 1 to the total number of timepoints
#'
#' @examples
#' \donttest{
#' # Create a sampling frame
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
#' s <- samples(sf)
#' length(s) # 220
#' range(s) # c(1, 220)
#' }
#'
#' @seealso
#' \code{\link{n_timepoints}} for total number of samples,
#' \code{\link{blockids}} for run membership
#' @export
samples <- function(x, ...) {
  UseMethod("samples")
}


#' Get Total Duration from Sampling Frame
#'
#' Calculates the total duration of the fMRI acquisition in seconds
#' across all runs.
#'
#' @param x An object containing temporal structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @return Numeric value representing total duration in seconds
#'
#' @examples
#' \donttest{
#' # Create a sampling frame: 220 timepoints with TR = 2 seconds
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
#' get_total_duration(sf) # Returns: 440 seconds
#' }
#'
#' @seealso
#' \code{\link{get_run_duration}} for individual run durations,
#' \code{\link{get_TR}} for repetition time
#' @export
get_total_duration <- function(x, ...) {
  UseMethod("get_total_duration")
}

#' Get Run Duration from Sampling Frame
#'
#' Calculates the duration of each run in seconds.
#'
#' @param x An object containing temporal structure (e.g., sampling_frame, fmri_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @return Numeric vector where each element represents the duration
#'   of the corresponding run in seconds
#'
#' @examples
#' \donttest{
#' # Create a sampling frame with different run lengths
#' sf <- fmrihrf::sampling_frame(blocklens = c(100, 120), TR = 2)
#' get_run_duration(sf) # Returns: c(200, 240) seconds
#' }
#'
#' @seealso
#' \code{\link{get_total_duration}} for total duration,
#' \code{\link{get_run_lengths}} for run lengths in timepoints
#' @export
get_run_duration <- function(x, ...) {
  UseMethod("get_run_duration")
}

#' Resolve Indices from Series Selector
#'
#' Converts a series selector specification into actual voxel indices
#' within the dataset mask.
#'
#' @param selector A series selector object (e.g., index_selector, voxel_selector)
#' @param dataset An fMRI dataset object providing spatial context
#' @param ... Additional arguments passed to methods
#'
#' @details
#' Series selectors provide various ways to specify spatial subsets of
#' fMRI data. This generic function resolves these specifications into
#' actual indices that can be used to extract data. Different selector
#' types support different selection methods:
#' \itemize{
#'   \item \code{index_selector}: Direct indices into masked data
#'   \item \code{voxel_selector}: 3D coordinates
#'   \item \code{roi_selector}: Region of interest masks
#'   \item \code{sphere_selector}: Spherical regions
#' }
#'
#' @return Integer vector of indices into the masked data
#'
#' @examples
#' \donttest{
#' # Example with index selector
#' sel <- index_selector(1:10)
#' # indices <- resolve_indices(sel, dataset)
#'
#' # Example with voxel coordinates
#' sel <- voxel_selector(cbind(x = 10, y = 20, z = 15))
#' # indices <- resolve_indices(sel, dataset)
#' }
#'
#' @seealso
#' \code{\link{series_selector}} for selector types,
#' \code{\link{fmri_series}} for using selectors to extract data
#' @export
resolve_indices <- function(selector, dataset, ...) {
  UseMethod("resolve_indices")
}

# ========================================================================
# Documentation
# ========================================================================
#
# These generics enable the modular file structure by providing clean
# interfaces between different components:
#
# - data_access.R implements get_data*, get_mask*, blocklens* methods
# - data_chunks.R implements data_chunks* methods
# - conversions.R implements as.matrix_dataset* methods
# - dataset_constructors.R provides the objects these generics operate on
# - print_methods.R provides specialized display methods
# - series_selector.R implements resolve_indices* methods
#
# All original fmrireg/fmridataset functionality is preserved while
# improving code organization and maintainability.
# ========================================================================
#' Get Subject IDs from Multi-Subject Dataset
#'
#' Generic function to extract subject identifiers from multi-subject
#' fMRI dataset objects.
#'
#' @param x A multi-subject dataset object (e.g., fmri_study_dataset)
#' @param ... Additional arguments passed to methods
#'
#' @details
#' Multi-subject datasets contain data from multiple participants. This
#' function extracts the subject identifiers associated with each dataset.
#' The order of subject IDs corresponds to the order of datasets.
#'
#' @return Character vector of subject identifiers
#'
#' @export
subject_ids <- function(x, ...) {
  UseMethod("subject_ids")
}
