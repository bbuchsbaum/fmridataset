#' Workflow and Performance Enhancements (Phase 3)
#'
#' Implements caching and simple parallelisation for BIDS operations.
#' These utilities build on the Phase 1 and Phase 2 facade functions.
#'
#' @name bids_facade_phase3
NULL

# ---------------------------------------------------------------------------
# Generic for clear_cache()
# ---------------------------------------------------------------------------
#' Clear cached BIDS queries
#'
#' Generic function used to clear cached results from a BIDS facade object.
#'
#' @param x Object
#' @param ... Additional arguments (unused)
#' @export
clear_cache <- function(x, ...) {
  UseMethod("clear_cache")
}

#' @export
clear_cache.bids_facade <- function(x, ...) {
  if (!is.null(x$cache) && is.environment(x$cache)) {
    rm(list = ls(envir = x$cache), envir = x$cache)
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# Enhanced discover() method with caching and parallel processing
# ---------------------------------------------------------------------------
#' @export
discover.bids_facade <- function(x, ...) {
  if (!is.null(x$cache) && exists("discovery", envir = x$cache)) {
    return(get("discovery", envir = x$cache))
  }

  check_package_available("bidser", "BIDS discovery", error = TRUE)

  funs <- list(
    summary = function() bidser::bids_summary(x$project),
    participants = function() bidser::participants(x$project),
    tasks = function() bidser::tasks(x$project),
    sessions = function() bidser::sessions(x$project),
    quality = function() tryCatch(bidser::check_func_scans(x$project),
                                  error = function(e) NULL)
  )

  if (.Platform$OS.type != "windows" && length(funs) > 1) {
    res_list <- parallel::mclapply(funs, function(f) f(), mc.cores = 2)
  } else {
    res_list <- lapply(funs, function(f) f())
  }

  res <- list(
    summary = res_list$summary,
    participants = res_list$participants,
    tasks = res_list$tasks,
    sessions = res_list$sessions,
    quality = res_list$quality
  )
  class(res) <- "bids_discovery_enhanced"

  if (!is.null(x$cache) && is.environment(x$cache)) {
    assign("discovery", res, envir = x$cache)
  }

  res
}

# ---------------------------------------------------------------------------
# Multi-subject dataset helper
# ---------------------------------------------------------------------------
#' Convert multiple subjects to datasets
#'
#' Convenience function to create a list of \code{fmri_dataset} objects for
#' several subjects in a BIDS project. Processing is performed in parallel
#' when possible.
#'
#' @param x A \code{bids_facade} object
#' @param subjects Character vector of subject IDs (must be non-empty)
#' @param ... Additional arguments passed to \code{as.fmri_dataset}
#' @return List of \code{fmri_dataset} objects
#' @export
bids_collect_datasets <- function(x, subjects, ...) {
  stopifnot(inherits(x, "bids_facade"))

  # Validate subjects input
  if (!is.character(subjects) || length(subjects) == 0) {
    stop("subjects must be a non-empty character vector")
  }
  if (.Platform$OS.type != "windows" && length(subjects) > 1) {
    parallel::mclapply(subjects, function(s) {
      as.fmri_dataset(x, subject_id = s, ...)
    }, mc.cores = min(2, length(subjects)))
  } else {
    lapply(subjects, function(s) as.fmri_dataset(x, subject_id = s, ...))
  }
}
