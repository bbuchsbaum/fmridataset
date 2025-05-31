#' Minimal Elegant BIDS Facade (Phase 1)
#'
#' Implements the first phase of the BIDS integration plan. This file
#' provides a thin wrapper around the `bidser` package with pleasant
#' printing and simple dataset creation.
#'
#' @name bids_facade_phase1
NULL

# ---------------------------------------------------------------------------
# Generic for discover()
# ---------------------------------------------------------------------------
#' Discover information about an object
#'
#' Generic function used for BIDS objects in this phase.
#' @param x Object
#' @param ... Additional arguments passed to methods
#' @export
#' @keywords internal
discover <- function(x, ...) {
  UseMethod("discover")
}

# ---------------------------------------------------------------------------
# bids() constructor
# ---------------------------------------------------------------------------
#' Open a BIDS project elegantly
#'
#' Creates a BIDS facade object by wrapping `bidser::bids_project()`
#' and providing pretty printing. Requires the `bidser` package.
#'
#' @param path Path to a BIDS dataset
#' @param ... Additional arguments passed to `bidser::bids_project`
#' @return An object of class `bids_facade`
#' @export
bids <- function(path, ...) {
  check_package_available("bidser", "BIDS access", error = TRUE)
  proj <- bidser::bids_project(path, ...)
  obj <- list(
    path = path,
    project = proj,
    cache = new.env(parent = emptyenv())
  )
  class(obj) <- "bids_facade"
  obj
}

#' @export
print.bids_facade <- function(x, ...) {
  cat("\u2728 Elegant BIDS Project\n")
  cat("Path:", x$path, "\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# discover() method
# ---------------------------------------------------------------------------
#' @keywords internal
discover_phase1.bids_facade <- function(x, ...) {
  check_package_available("bidser", "BIDS discovery", error = TRUE)
  res <- list(
    summary = bidser::bids_summary(x$project),
    participants = bidser::participants(x$project),
    tasks = bidser::tasks(x$project),
    sessions = bidser::sessions(x$project)
  )
  class(res) <- "bids_discovery_simple"
  res
}

#' @export
print.bids_discovery_simple <- function(x, ...) {
  cat("\u2728 BIDS Discovery\n")
  cat(length(x$participants$participant_id), "participants\n")
  cat(length(x$tasks$task_id), "tasks\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# as.fmri_dataset method
# ---------------------------------------------------------------------------
#' @export
as.fmri_dataset.bids_facade <- function(x, ...) {
  as.fmri_dataset(x$project, ...)
}
