#' S3 Generics for fmridataset Package
#'
#' This file defines S3 generic functions used throughout the fmridataset package.
#' The 'aaa_' prefix ensures this file is loaded first, establishing the generics
#' before their methods are defined in other files.
#'
#' @name generics
NULL

#' Convert Objects to fmri_dataset
#'
#' Generic function to convert various input types to `fmri_dataset` objects.
#' This provides a unified interface for creating `fmri_dataset` objects from
#' different data sources including file paths, pre-loaded objects, matrices,
#' and BIDS projects.
#'
#' @param x Object to convert to `fmri_dataset`
#' @param ... Additional arguments passed to specific methods
#' @return An `fmri_dataset` object
#' @export
#' @family fmri_dataset
#' @seealso \code{\link{fmri_dataset_create}} for the primary constructor
as.fmri_dataset <- function(x, ...) {
  UseMethod("as.fmri_dataset")
}

#' Default Method for as.fmri_dataset
#'
#' @param x Object that cannot be converted
#' @param ... Additional arguments (ignored)
#' @return Throws an error
#' @export
as.fmri_dataset.default <- function(x, ...) {
  stop("Cannot convert object of class '", class(x)[1], "' to fmri_dataset.\n",
       "Supported types: character (file paths), list (NeuroVec objects), ",
       "matrix/array (data matrix), bids_project")
}

#' Add Subject Filter to BIDS Query
#'
#' Generic function for adding subject filters to BIDS queries.
#'
#' @param x Object to add subject filter to
#' @param ... Subject IDs to include
#' @return Modified object (for chaining)
#' @export
subject <- function(x, ...) {
  UseMethod("subject")
}

#' Add Task Filter to BIDS Query
#'
#' Generic function for adding task filters to BIDS queries.
#'
#' @param x Object to add task filter to
#' @param ... Task names to include
#' @return Modified object (for chaining)
#' @export
task <- function(x, ...) {
  UseMethod("task")
}

#' Add Session Filter to BIDS Query
#'
#' Generic function for adding session filters to BIDS queries.
#'
#' @param x Object to add session filter to
#' @param ... Session IDs to include
#' @return Modified object (for chaining)
#' @export
session <- function(x, ...) {
  UseMethod("session")
}

#' Add Run Filter to BIDS Query
#'
#' Generic function for adding run filters to BIDS queries.
#'
#' @param x Object to add run filter to
#' @param ... Run numbers to include
#' @return Modified object (for chaining)
#' @export
run <- function(x, ...) {
  UseMethod("run")
}

#' Add Derivatives Filter to BIDS Query
#'
#' Generic function for adding derivatives filters to BIDS queries.
#'
#' @param x Object to add derivatives filter to
#' @param ... Derivative pipeline names to include
#' @return Modified object (for chaining)
#' @export
derivatives <- function(x, ...) {
  UseMethod("derivatives")
}

#' Add Space Filter to BIDS Query
#'
#' Generic function for adding space filters to BIDS queries.
#'
#' @param x Object to add space filter to
#' @param ... Space names to include (for derivatives)
#' @return Modified object (for chaining)
#' @export
space <- function(x, ...) {


  UseMethod("space")
}
#' Discover details about an object
#'
#' Generic used for BIDS facades
#' @param x Object
#' @param ... Additional args
#' @export
discover <- function(x, ...) {
  UseMethod("discover")
}

#' Focus on specific task
#'
#' Natural language verb for selecting a task of interest.
#' @param x Object
#' @param ... Task identifiers
#' @return Modified object (for chaining)
#' @export
focus_on <- function(x, ...) {
  UseMethod("focus_on")
}

#' Filter for young adult participants
#'
#' Natural language verb for selecting young adults (approx. 18-35).
#' @param x Object
#' @param ... Additional arguments
#' @return Modified object (for chaining)
#' @export
from_young_adults <- function(x, ...) {
  UseMethod("from_young_adults")
}

#' Require excellent quality data
#'
#' Natural language verb for enforcing high quality thresholds.
#' @param x Object
#' @param ... Additional arguments
#' @return Modified object (for chaining)
#' @export
with_excellent_quality <- function(x, ...) {
  UseMethod("with_excellent_quality")
}

#' Specify preprocessing pipeline
#'
#' Natural language verb for choosing a preprocessing pipeline.
#' @param x Object
#' @param ... Pipeline name
#' @return Modified object (for chaining)
#' @export
preprocessed_with <- function(x, ...) {
  UseMethod("preprocessed_with")
}

#' Narratively describe a BIDS project
#'
#' Provides a short human-friendly summary.
#' @param x Object
#' @param ... Additional arguments
#' @return Invisible result
#' @export
tell_me_about <- function(x, ...) {
  UseMethod("tell_me_about")
}

#' Create dataset from conversational chain
#'
#' Final verb that converts the chain into an `fmri_dataset`.
#' @param x Object
#' @param ... Additional arguments passed to dataset creation
#' @return fmri_dataset object
#' @export
create_dataset <- function(x, ...) {
  UseMethod("create_dataset")
}

