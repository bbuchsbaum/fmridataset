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
#' @export
from_young_adults.bids_facade <- function(x, ...) {
  if (is.null(x$nl_filters)) x$nl_filters <- list()
  x$nl_filters$age_range <- c(18, 35)
  x
}

# ---------------------------------------------------------------------------
# with_excellent_quality() - quality filter
# ---------------------------------------------------------------------------
#' @export
with_excellent_quality.bids_facade <- function(x, ...) {
  if (is.null(x$nl_filters)) x$nl_filters <- list()
  x$nl_filters$quality <- "excellent"
  x
}

# ---------------------------------------------------------------------------
# preprocessed_with() - choose pipeline
# ---------------------------------------------------------------------------
#' @export
preprocessed_with.bids_facade <- function(x, pipeline, ...) {
  if (is.null(x$nl_filters)) x$nl_filters <- list()
  x$nl_filters$pipeline <- pipeline
  x
}

# ---------------------------------------------------------------------------
# tell_me_about() - narrative summary
# ---------------------------------------------------------------------------
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

