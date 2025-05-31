#' Enhanced BIDS Facade (Phase 2)
#'
#' Implements features from Phase 2 of the BIDS integration plan.
#' Provides enhanced discovery output and basic quality assessment
#' utilities built on top of the `bidser` package.
#'
#' @name bids_facade_phase2
NULL

# ---------------------------------------------------------------------------
# Generic for assess_quality()
# ---------------------------------------------------------------------------
#' Assess quality of a BIDS project
#'
#' Generic for quality assessment methods.
#' @param x Object
#' @param ... Additional arguments passed to methods
#' @export
#' @keywords internal
assess_quality <- function(x, ...) {
  UseMethod("assess_quality")
}

# ---------------------------------------------------------------------------
# Enhanced discover() method with quality metrics
# ---------------------------------------------------------------------------
#' @keywords internal
discover_phase2.bids_facade <- function(x, ...) {
  check_package_available("bidser", "BIDS discovery", error = TRUE)

  summary_tbl <- bidser::bids_summary(x$project)
  part_tbl <- bidser::participants(x$project)
  task_tbl <- bidser::tasks(x$project)
  sess_tbl <- bidser::sessions(x$project)
  q_metrics <- tryCatch(
    bidser::check_func_scans(x$project),
    error = function(e) NULL
  )

  res <- list(
    summary = summary_tbl,
    participants = part_tbl,
    tasks = task_tbl,
    sessions = sess_tbl,
    quality = q_metrics
  )
  class(res) <- "bids_discovery_enhanced"
  res
}

#' @export
print.bids_discovery_enhanced <- function(x, ...) {
  cat("\u2728 BIDS Discovery\n")
  part_count <- if (is.data.frame(x$participants)) {
    nrow(x$participants)
  } else {
    length(x$participants)
  }
  task_count <- if (is.data.frame(x$tasks)) {
    nrow(x$tasks)
  } else {
    length(x$tasks)
  }
  cat(part_count, "participants\n")
  cat(task_count, "tasks\n")
  if (!is.null(x$quality)) {
    cat("Quality metrics available\n")
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# assess_quality() method
# ---------------------------------------------------------------------------
#' @export
assess_quality.bids_facade <- function(x, subject_id, session_id = NULL,
                                       task_id = NULL, run_ids = NULL) {
  check_package_available("bidser", "quality assessment", error = TRUE)

  confounds <- tryCatch(
    bidser::read_confounds(x$project,
                           subject_id = subject_id,
                           session_id = session_id,
                           task_id = task_id,
                           run_ids = run_ids),
    error = function(e) NULL
  )

  scans <- tryCatch(
    bidser::func_scans(x$project,
                       subject_id = subject_id,
                       session_id = session_id,
                       task_id = task_id,
                       run_ids = run_ids),
    error = function(e) {
      warning("Could not retrieve functional scans: ", conditionMessage(e))
      NULL
    }
  )

  metrics <- if (is.null(scans)) {
    NULL
  } else {
    tryCatch(bidser::check_func_scans(scans),
             error = function(e) NULL)
  }

  mask <- tryCatch(
    bidser::create_preproc_mask(x$project,
                                subject_id = subject_id,
                                session_id = session_id),
    error = function(e) NULL
  )

  res <- list(confounds = confounds,
              quality_metrics = metrics,
              mask = mask)
  class(res) <- "bids_quality_report"
  res
}

#' @export
print.bids_quality_report <- function(x, ...) {
  cat("\u2728 BIDS Quality Report\n")
  if (!is.null(x$quality_metrics)) {
    cat(nrow(x$quality_metrics), "scan checks\n")
  }
  if (!is.null(x$confounds)) {
    cat(ncol(x$confounds), "confound regressors\n")
  }
  invisible(x)
}
