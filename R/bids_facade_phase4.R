#' Conversational BIDS Verbs (Phase 4)
#'
#' Implements simple natural language verbs that build on the
#' previous BIDS facade phases. These helpers allow expressive
#' method chaining when interacting with BIDS datasets.
#'
#' @name bids_facade_phase4
NULL

# ---------------------------------------------------------------------------
# focus_on() - select task
# ---------------------------------------------------------------------------
#' Focus on specific tasks
#'
#' Adds a task filter to a BIDS facade object.
#'
#' @param x A `bids_facade` object
#' @param ... Character task identifiers
#' @return Modified `bids_facade` object for further chaining
#' @examples
#' \dontrun{
#' bids("path") %>% focus_on("rest")
#' }
#' @export
focus_on.bids_facade <- function(x, ...) {
  tasks <- c(...)
  if (is.null(x$nl_filters)) x$nl_filters <- list()
  x$nl_filters$task <- tasks
  x
}

# ---------------------------------------------------------------------------
# from_young_adults() - demographic filter
# ---------------------------------------------------------------------------
#' Filter for young adult participants
#'
#' Sets an age range filter of approximately 18-35 years.
#'
#' @param x A `bids_facade` object
#' @param ... Additional arguments (unused)
#' @return Modified `bids_facade` object for further chaining
#' @examples
#' \dontrun{
#' bids("path") %>% from_young_adults()
#' }
#' @export
from_young_adults.bids_facade <- function(x, ...) {
  if (is.null(x$nl_filters)) x$nl_filters <- list()
  x$nl_filters$age_range <- c(18, 35)
  x
}

# ---------------------------------------------------------------------------
# with_excellent_quality() - quality filter
# ---------------------------------------------------------------------------
#' Require excellent data quality
#'
#' Adds a quality filter marking only "excellent" scans.
#'
#' @param x A `bids_facade` object
#' @param ... Additional arguments (unused)
#' @return Modified `bids_facade` object for further chaining
#' @examples
#' \dontrun{
#' bids("path") %>% with_excellent_quality()
#' }
#' @export
with_excellent_quality.bids_facade <- function(x, ...) {
  if (is.null(x$nl_filters)) x$nl_filters <- list()
  x$nl_filters$quality <- "excellent"
  x
}

# ---------------------------------------------------------------------------
# preprocessed_with() - choose pipeline
# ---------------------------------------------------------------------------
#' Specify preprocessing pipeline
#'
#' Records the desired preprocessing pipeline name.
#'
#' @param x A `bids_facade` object
#' @param pipeline Character name of the preprocessing pipeline
#' @param ... Additional arguments (unused)
#' @return Modified `bids_facade` object for further chaining
#' @examples
#' \dontrun{
#' bids("path") %>% preprocessed_with("fmriprep")
#' }
#' @export
preprocessed_with.bids_facade <- function(x, pipeline, ...) {
  if (is.null(x$nl_filters)) x$nl_filters <- list()
  x$nl_filters$pipeline <- pipeline
  x
}

# ---------------------------------------------------------------------------
# tell_me_about() - narrative summary
# ---------------------------------------------------------------------------
#' Provide a narrative dataset summary
#'
#' Prints a short human-friendly description of the dataset and
#' invisibly returns discovery information.
#'
#' @param x A `bids_facade` object
#' @param ... Additional arguments passed to `discover`
#' @return Invisibly returns a discovery list or `NULL` when unavailable
#' @examples
#' \dontrun{
#' bids("path") %>% tell_me_about()
#' }
#' @export
tell_me_about.bids_facade <- function(x, ...) {
  info <- tryCatch(discover(x), error = function(e) NULL)
  cat("\U1F3AD The Story of Your Data\n")
  if (!is.null(info)) {
    if (!is.null(info$summary)) {
      cat("Subjects:", nrow(info$participants), "\n")
      cat("Tasks:", nrow(info$tasks), "\n")
    }
  } else {
    cat("No information available.\n")
  }
  invisible(info)
}

# ---------------------------------------------------------------------------
# create_dataset() - finalise chain
# ---------------------------------------------------------------------------
#' Create a dataset from the conversational chain
#'
#' Converts the chained filters into an `fmri_dataset` object.
#'
#' @param x A `bids_facade` object
#' @param subject_id Character subject identifier
#' @param ... Additional arguments passed to `as.fmri_dataset`
#' @return An `fmri_dataset` object
#' @examples
#' \dontrun{
#' bids("path") %>% focus_on("rest") %>% create_dataset("sub-01")
#' }
#' @export
create_dataset.bids_facade <- function(x, subject_id, ...) {
  task_id <- NULL
  image_type <- "auto"
  if (!is.null(x$nl_filters)) {
    task_id <- x$nl_filters$task %||% NULL
    if (!is.null(x$nl_filters$pipeline)) {
      image_type <- x$nl_filters$pipeline
    }
  }
  as.fmri_dataset(x$project,
                  subject_id = subject_id,
                  task_id = task_id,
                  image_type = image_type,
                  ...)
}

