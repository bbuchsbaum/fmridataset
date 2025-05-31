#' AI & Community Integration (Phase 5)
#'
#' Implements experimental workflow helpers and community wisdom
#' utilities as described in Phase 5 of the BIDS integration plan.
#' These functions provide simple infrastructure for creating
#' shareable workflows and retrieving best-practice tips.
#'
#' @name bids_facade_phase5
NULL

# ---------------------------------------------------------------------------
# create_workflow() constructor
# ---------------------------------------------------------------------------
#' Create a shareable analysis workflow
#'
#' Constructs a simple workflow object that can be populated with
#' processing steps and applied to `fmri_dataset` objects. This is a
#' lightweight representation used to demonstrate the Phase 5 design.
#'
#' @param name Character name of the workflow
#' @return A `bids_workflow` object
#' @export
create_workflow <- function(name, ...) {
  UseMethod("create_workflow")
}

#' @export
create_workflow.default <- function(name, ...) {
  wf <- list(
    name = name,
    description = NULL,
    steps = list(),
    finished = FALSE
  )
  class(wf) <- "bids_workflow"
  wf
}

# ---------------------------------------------------------------------------
# describe() method
# ---------------------------------------------------------------------------
#' @export
describe.bids_workflow <- function(x, text, ...) {
  x$description <- text
  x
}

# ---------------------------------------------------------------------------
# add_step() method
# ---------------------------------------------------------------------------
#' @export
add_step.bids_workflow <- function(x, step, ...) {
  x$steps <- append(x$steps, list(step))
  x
}

# ---------------------------------------------------------------------------
# finish_with_flourish() method
# ---------------------------------------------------------------------------
#' @export
finish_with_flourish.bids_workflow <- function(x, ...) {
  x$finished <- TRUE
  x
}

# ---------------------------------------------------------------------------
# apply_workflow() method for fmri_dataset
# ---------------------------------------------------------------------------
#' Apply a workflow to a dataset
#'
#' Iteratively applies each step function of a workflow to an
#' `fmri_dataset` object. Steps that are not functions are ignored.
#'
#' @param x An `fmri_dataset` object
#' @param workflow A `bids_workflow` object
#' @return Modified `fmri_dataset` object
#' @export
apply_workflow <- function(x, ...) {
  UseMethod("apply_workflow")
}

#' @export
apply_workflow.fmri_dataset <- function(x, workflow, ...) {
  stopifnot(inherits(workflow, "bids_workflow"))
  dset <- x
  for (st in workflow$steps) {
    if (is.function(st)) {
      dset <- st(dset)
    }
  }
  dset
}

# ---------------------------------------------------------------------------
# discover_best_practices() utility
# ---------------------------------------------------------------------------
#' Retrieve community best practices
#'
#' Returns a small text snippet with tips for the given topic. This is a
#' placeholder demonstrating how community wisdom could be surfaced.
#'
#' @param topic Character topic name (e.g., "motion_correction")
#' @return Character message with best-practice advice
#' @export
discover_best_practices <- function(topic = "general", ...) {
  msg <- switch(
    topic,
    motion_correction = paste(
      "\U2728 Community Wisdom for Motion Correction:\n",
      " - FD threshold: 0.2mm\n",
      " - DVARS threshold: 75th percentile\n",
      " - Scrubbing with interpolation\n",
      " - Confound regression: motion_24 + compcor"
    ),
    paste("No community advice for", topic)
  )
  cat(msg, "\n")
  invisible(msg)
}
