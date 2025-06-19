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
#' @param x An fMRI dataset object
#' @param ... Additional arguments passed to methods
#' @return Dataset-specific data object
#' @export
get_data <- function(x, ...) {
  UseMethod("get_data")
}

#' Get Data Matrix from fMRI Dataset Objects
#'
#' Generic function to extract data as a matrix from various fMRI dataset types.
#' Always returns a matrix with timepoints as rows and voxels as columns.
#'
#' @param x An fMRI dataset object
#' @param ... Additional arguments passed to methods
#' @return A matrix with timepoints as rows and voxels as columns
#' @export
get_data_matrix <- function(x, ...) {
  UseMethod("get_data_matrix")
}

#' Get Mask from fMRI Dataset Objects
#'
#' Generic function to extract masks from various fMRI dataset types.
#' Returns the mask in its appropriate format for the dataset type.
#'
#' @param x An fMRI dataset object
#' @param ... Additional arguments passed to methods
#' @return Mask object (NeuroVol, vector, etc.)
#' @export
get_mask <- function(x, ...) {
  UseMethod("get_mask")
}

#' Get Block Lengths from Objects
#'
#' Generic function to extract block/run lengths from various objects.
#' Extends the sampling_frame generic to work with dataset objects.
#'
#' @param x An object with block structure
#' @param ... Additional arguments passed to methods
#' @return Integer vector of block/run lengths
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
#' @return A chunk iterator object
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
#' @return A matrix_dataset object
#' @export
as.matrix_dataset <- function(x, ...) {
  UseMethod("as.matrix_dataset")
}

# Sampling frame generics
#' Get TR from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
get_TR <- function(x, ...) {
  UseMethod("get_TR")
}

#' Get run lengths from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
get_run_lengths <- function(x, ...) {
  UseMethod("get_run_lengths")
}

#' Get number of runs from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
n_runs <- function(x, ...) {
  UseMethod("n_runs")
}

#' Get number of timepoints from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
n_timepoints <- function(x, ...) {
  UseMethod("n_timepoints")
}

#' Get block IDs from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
blockids <- function(x, ...) {
  UseMethod("blockids")
}

#' Get samples from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
samples <- function(x, ...) {
  UseMethod("samples")
}

#' Get global onsets from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
global_onsets <- function(x, ...) {
  UseMethod("global_onsets")
}

#' Get total duration from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
get_total_duration <- function(x, ...) {
  UseMethod("get_total_duration")
}

#' Get run duration from sampling frame
#' @param x Sampling frame object
#' @param ... Additional arguments
#' @export
get_run_duration <- function(x, ...) {
  UseMethod("get_run_duration")
}

#' Resolve indices from series selector
#' @param selector Series selector object
#' @param dataset Dataset object for context
#' @param ... Additional arguments
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
